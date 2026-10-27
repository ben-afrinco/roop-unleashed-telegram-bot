#!/usr/bin/env python3
"""Telegram UI for roop-unleashed.

The bot is compatible with local Python and Google Colab. Jobs are serialized
because roop keeps model and processing state in module-level globals.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Colab exports matplotlib's notebook backend, which is not available inside
# the isolated Python 3.10 environment used by this bot.
os.environ.setdefault("MPLBACKEND", "Agg")

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import roop.core as core
import roop.globals
import roop.utilities as util
from settings import Settings

LOGGER = logging.getLogger("roop.telegram")
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_ROOT = PROJECT_ROOT / "telegram_data"
MAX_DOWNLOAD_BYTES = int(os.getenv("TELEGRAM_MAX_FILE_MB", "500")) * 1024 * 1024
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEOS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}


@dataclass
class UserSession:
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None
    many_faces: bool = False
    skip_audio: bool = False
    keep_fps: bool = True
    busy: bool = False
    cancel_requested: bool = False
    work_dir: Optional[Path] = None
    current_job: Optional["Job"] = None


@dataclass
class Job:
    chat_id: int
    user_id: int
    source_path: Path
    target_path: Path
    work_dir: Path
    many_faces: bool
    skip_audio: bool
    keep_fps: bool
    future: asyncio.Future[Path]
    current: int = 0
    total: int = 0
    status: str = "في طابور الانتظار"
    cancelled: bool = False

    def report(self, current: int = -1, total: int = -1, status: str = "") -> None:
        if current >= 0:
            self.current = current
        if total >= 0:
            self.total = total
        if status:
            self.status = status


class TelegramRoopBot:
    def __init__(self, token: str, provider: str = "cpu") -> None:
        self.token = token
        self.provider = provider
        self.sessions: dict[int, UserSession] = {}
        self.jobs: asyncio.Queue[Job] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task[None]] = None
        self.active_job: Optional[Job] = None

    def session(self, user_id: int) -> UserSession:
        return self.sessions.setdefault(user_id, UserSession())

    async def start(self, application: Application) -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting Telegram bot with provider=%s", self.provider)
        if not core.pre_check():
            raise RuntimeError("roop pre-check failed")
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists() and (PROJECT_ROOT / "config_colab.yaml").exists():
            shutil.copy2(PROJECT_ROOT / "config_colab.yaml", config_path)
        roop.globals.CFG = Settings(str(config_path))
        self.worker_task = asyncio.create_task(self.process_queue(), name="roop-worker")

    async def stop(self, application: Application) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
        for session in self.sessions.values():
            if session.work_dir:
                shutil.rmtree(session.work_dir, ignore_errors=True)

    async def process_queue(self) -> None:
        while True:
            job = await self.jobs.get()
            self.active_job = job
            try:
                if job.cancelled:
                    raise RuntimeError("تم إلغاء المهمة قبل بدء المعالجة")
                result = await asyncio.to_thread(self.process_job, job)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                LOGGER.exception("Telegram job failed")
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.active_job = None
                self.jobs.task_done()

    def process_job(self, job: Job) -> Path:
        output_dir = job.work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        roop.globals.source_path = str(job.source_path)
        roop.globals.target_path = str(job.target_path)
        roop.globals.output_path = str(output_dir / "result")
        roop.globals.target_folder_path = None
        roop.globals.headless = True
        roop.globals.frame_processors = ["face_swapper"]
        roop.globals.keep_fps = job.keep_fps
        roop.globals.keep_frames = False
        roop.globals.skip_audio = job.skip_audio
        roop.globals.many_faces = job.many_faces
        roop.globals.source_face_index = 0
        roop.globals.target_face_index = 0
        roop.globals.video_encoder = "libx264"
        roop.globals.video_quality = 18
        roop.globals.max_memory = 0
        roop.globals.execution_threads = max(1, min(os.cpu_count() or 1, 4))
        core.set_execution_provider(self.provider)
        roop.globals.INPUT_FACES = []
        roop.globals.TARGET_FACES = []
        roop.globals.processing = False
        roop.globals.progress_callback = job.report
        try:
            job.report(status="فحص الوجوه")
            source_faces = core.extract_face_images(str(job.source_path), (False, 0))
            if not source_faces:
                raise ValueError("لم يتم العثور على وجه في صورة المصدر")
            roop.globals.INPUT_FACES.append(source_faces[0])
            target_faces = core.extract_face_images(
                str(job.target_path), (False, util.has_image_extension(str(job.target_path)))
            )
            if not target_faces:
                raise ValueError("لم يتم العثور على وجه في الهدف")
            roop.globals.TARGET_FACES.append(target_faces[0])
            job.report(status="بدأت المعالجة")
            core.batch_process(None, False, None, True)
        finally:
            roop.globals.processing = False
            roop.globals.progress_callback = None
            roop.globals.INPUT_FACES = []
            roop.globals.TARGET_FACES = []
            core.release_resources()
        candidates = [p for p in output_dir.iterdir() if p.is_file()]
        if not candidates:
            raise RuntimeError("لم ينتج محرك roop ملفًا قابلًا للإرسال")
        job.report(current=job.total or job.current, status="اكتملت المعالجة")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    async def enqueue(self, job: Job) -> Path:
        await self.jobs.put(job)
        return await job.future


def get_bot(context: ContextTypes.DEFAULT_TYPE) -> TelegramRoopBot:
    bot = context.application.bot_data.get("roop_bot")
    if not isinstance(bot, TelegramRoopBot):
        raise RuntimeError("bot is not initialized")
    return bot


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def settings_text(session: UserSession) -> str:
    return (
        "إعدادات الجلسة:\n"
        f"• تبديل كل الوجوه: {'مفعّل' if session.many_faces else 'معطّل'}\n"
        f"• تجاهل صوت الفيديو: {'مفعّل' if session.skip_audio else 'معطّل'}\n"
        f"• الحفاظ على FPS: {'مفعّل' if session.keep_fps else 'معطّل'}"
    )


def main_keyboard(session: UserSession, processing: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("الإعدادات", callback_data="settings"), InlineKeyboardButton("الحالة", callback_data="status")],
        [InlineKeyboardButton("كل الوجوه", callback_data="toggle:many_faces"), InlineKeyboardButton("تجاهل الصوت", callback_data="toggle:skip_audio")],
        [InlineKeyboardButton("الحفاظ على FPS", callback_data="toggle:keep_fps")],
    ]
    if processing:
        rows.append([InlineKeyboardButton("إلغاء المعالجة", callback_data="cancel")])
    else:
        rows.append([InlineKeyboardButton("إعادة ضبط الجلسة", callback_data="reset")])
    return InlineKeyboardMarkup(rows)


def progress_text(job: Job) -> str:
    if job.total > 0:
        done = min(job.current, job.total)
        percent = int(done * 100 / job.total)
        filled = int(percent / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{job.status}\n[{bar}] {percent}% ({done}/{job.total} إطار)"
    return f"{job.status}\nجاري التحضير..."


def extension_for(filename: str, fallback: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in SUPPORTED_IMAGES | SUPPORTED_VIDEOS else fallback


async def download_media(update: Update, session: UserSession, media, filename: str, fallback: str) -> Path:
    if getattr(media, "file_size", None) and media.file_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"حجم الملف يتجاوز {human_size(MAX_DOWNLOAD_BYTES)}")
    suffix = extension_for(filename, fallback)
    if suffix not in SUPPORTED_IMAGES | SUPPORTED_VIDEOS:
        raise ValueError("الامتدادات المدعومة: JPG, PNG, WEBP, BMP, MP4, MOV, MKV, AVI, WEBM, GIF")
    if session.work_dir is None:
        session.work_dir = Path(tempfile.mkdtemp(prefix=f"user_{update.effective_user.id}_", dir=WORK_ROOT))
    name = "source" if session.source_path is None else "target"
    destination = session.work_dir / f"{name}{suffix}"
    telegram_file = await media.get_file()
    await telegram_file.download_to_drive(custom_path=str(destination))
    if destination.stat().st_size > MAX_DOWNLOAD_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError(f"حجم الملف يتجاوز {human_size(MAX_DOWNLOAD_BYTES)}")
    return destination


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_bot(context).session(update.effective_user.id)
    if not session.busy:
        session.source_path = session.target_path = None
    await update.message.reply_text(
        "مرحبًا بك في بوت roop.\nأرسل صورة الوجه المصدر، ثم أرسل الصورة أو الفيديو الهدف.\n\n" + settings_text(session),
        reply_markup=main_keyboard(session, session.busy),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "طريقة الاستخدام:\n1) أرسل صورة المصدر.\n2) اضغط الإعدادات إن أردت تغييرها.\n3) أرسل الهدف؛ تبدأ المعالجة تلقائيًا.\n4) ستظهر نسبة الإطارات مباشرة وتصل النتيجة تلقائيًا.\n\n"
        "يمكنك أيضًا استعمال /start و /settings و /status و /reset و /cancel."
    )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_bot(context).session(update.effective_user.id)
    await update.message.reply_text(settings_text(session), reply_markup=main_keyboard(session, session.busy))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = get_bot(context)
    session = service.session(update.effective_user.id)
    text = f"المصدر: {'جاهز' if session.source_path else 'لم يُرسل'}\nالهدف: {'جاهز' if session.target_path else 'لم يُرسل'}\nالحالة: {'قيد المعالجة' if session.busy else 'جاهز'}\nالمهام في الطابور: {service.jobs.qsize()}"
    if session.current_job:
        text += "\n\n" + progress_text(session.current_job)
    await update.message.reply_text(text, reply_markup=main_keyboard(session, session.busy))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_bot(context).session(update.effective_user.id)
    if session.busy:
        await update.message.reply_text("لا يمكن إعادة الضبط أثناء المعالجة؛ استخدم زر الإلغاء أولًا.")
        return
    if session.work_dir:
        shutil.rmtree(session.work_dir, ignore_errors=True)
    session.source_path = session.target_path = session.work_dir = None
    session.cancel_requested = False
    await update.message.reply_text("تمت إعادة ضبط الجلسة. أرسل صورة المصدر.", reply_markup=main_keyboard(session))


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = get_bot(context)
    session = service.session(update.effective_user.id)
    if not session.busy:
        await update.message.reply_text("لا توجد معالجة جارية.")
        return
    session.cancel_requested = True
    if session.current_job:
        session.current_job.cancelled = True
    if service.active_job and service.active_job.user_id == update.effective_user.id:
        roop.globals.processing = False
    await update.message.reply_text("تم إرسال طلب الإلغاء. ستتوقف المعالجة عند أقرب نقطة آمنة.")


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    service = get_bot(context)
    session = service.session(update.effective_user.id)
    action = query.data or ""
    if action.startswith("toggle:"):
        field = action.split(":", 1)[1]
        setattr(session, field, not getattr(session, field))
        await query.edit_message_text(settings_text(session), reply_markup=main_keyboard(session, session.busy))
    elif action == "settings":
        await query.edit_message_text(settings_text(session), reply_markup=main_keyboard(session, session.busy))
    elif action == "status":
        text = f"الحالة: {'قيد المعالجة' if session.busy else 'جاهز'}\nالمهام في الطابور: {service.jobs.qsize()}"
        if session.current_job:
            text += "\n\n" + progress_text(session.current_job)
        await query.edit_message_text(text, reply_markup=main_keyboard(session, session.busy))
    elif action == "reset":
        if session.busy:
            await query.edit_message_text("لا يمكن إعادة الضبط أثناء المعالجة؛ استخدم زر الإلغاء أولًا.", reply_markup=main_keyboard(session, True))
        else:
            if session.work_dir:
                shutil.rmtree(session.work_dir, ignore_errors=True)
            session.source_path = session.target_path = session.work_dir = None
            session.cancel_requested = False
            await query.edit_message_text("تمت إعادة ضبط الجلسة. أرسل صورة المصدر.", reply_markup=main_keyboard(session))
    elif action == "cancel":
        await cmd_cancel(update, context)


async def receive_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = get_bot(context)
    session = service.session(update.effective_user.id)
    if session.busy:
        await update.message.reply_text("لديك مهمة قيد المعالجة. استخدم زر الحالة أو الإلغاء.")
        return
    message = update.message
    try:
        if message.photo:
            media, filename, fallback = message.photo[-1], "photo.jpg", ".jpg"
        elif message.video:
            media, filename, fallback = message.video, message.video.file_name or "video.mp4", ".mp4"
        elif message.document:
            media, filename, fallback = message.document, message.document.file_name or "file", ".bin"
        else:
            return
        await message.chat.send_action(ChatAction.TYPING)
        if session.source_path is None:
            source = await download_media(update, session, media, filename, fallback)
            if not util.is_image(str(source)):
                source.unlink(missing_ok=True)
                raise ValueError("المصدر يجب أن يكون صورة")
            session.source_path = source
            await message.reply_text("تم حفظ المصدر. أرسل الآن الهدف.", reply_markup=main_keyboard(session))
            return
        target = await download_media(update, session, media, filename, fallback)
        if not util.is_image(str(target)) and not util.is_video(str(target)) and target.suffix.lower() != ".gif":
            target.unlink(missing_ok=True)
            raise ValueError("الهدف يجب أن يكون صورة أو فيديو مدعومًا")
        session.target_path = target
        session.busy = True
        session.cancel_requested = False
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Path] = loop.create_future()
        job = Job(update.effective_chat.id, update.effective_user.id, session.source_path, target, session.work_dir, session.many_faces, session.skip_audio, session.keep_fps, future)
        session.current_job = job
        status_message = await message.reply_text(progress_text(job), reply_markup=main_keyboard(session, True))
        monitor = asyncio.create_task(monitor_progress(status_message, job, session), name=f"progress-{job.user_id}")
        try:
            result = await service.enqueue(job)
            await status_message.edit_text("اكتملت المعالجة، جاري إرسال النتيجة...", reply_markup=main_keyboard(session, False))
            if result.stat().st_size > MAX_DOWNLOAD_BYTES:
                await message.reply_text(f"اكتملت المعالجة، لكن الناتج حجمه {human_size(result.stat().st_size)} ويتجاوز حد الإرسال.")
            else:
                await message.reply_document(document=str(result), caption="اكتملت المعالجة.")
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
    except Exception as exc:
        LOGGER.exception("Media handling failed")
        await message.reply_text(f"تعذر إكمال الطلب: {exc}")
    finally:
        session.busy = False
        session.target_path = None
        session.current_job = None
        if session.work_dir and session.source_path:
            source_name = session.source_path.name
            for child in list(session.work_dir.iterdir()):
                if child.name != source_name:
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)


async def monitor_progress(message, job: Job, session: UserSession) -> None:
    last = ""
    while True:
        text = progress_text(job)
        if text != last:
            try:
                await message.edit_text(text, reply_markup=main_keyboard(session, True))
                last = text
            except Exception as exc:
                LOGGER.debug("Progress edit skipped: %s", exc)
        await asyncio.sleep(1.5)


async def post_init(application: Application) -> None:
    await application.bot_data["roop_bot"].start(application)


async def post_shutdown(application: Application) -> None:
    await application.bot_data["roop_bot"].stop(application)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run roop-unleashed through Telegram")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--provider", default=os.getenv("ROOP_EXECUTION_PROVIDER", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN or pass --token")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = TelegramRoopBot(args.token, args.provider)
    application = ApplicationBuilder().token(args.token).post_init(post_init).post_shutdown(post_shutdown).build()
    application.bot_data["roop_bot"] = service
    for command, handler in (("start", cmd_start), ("help", cmd_help), ("settings", cmd_settings), ("status", cmd_status), ("reset", cmd_reset), ("cancel", cmd_cancel)):
        application.add_handler(CommandHandler(command, handler))
    application.add_handler(CallbackQueryHandler(on_button))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, receive_media))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

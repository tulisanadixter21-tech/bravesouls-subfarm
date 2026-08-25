"""
BBS Sub-Story Auto-Farmer - GUI
--------------------------------
A simple control panel around subfarm_bot's logic: Start/Pause/Stop
buttons, a live log window, and a status indicator, instead of
watching a raw terminal.

STARTING POINT: put the game on the "Sub Stories" category grid screen
(Back button, "Sub Stories" title, category tiles like The Human World /
Soul Society / Hueco Mundo / Side Stories / Others) before you hit Start.
The bot assumes that's where it begins.

FARMING LOOP:
  Sub Stories grid -> click a NEW category -> click a NEW story tile ->
  Chapter detail -> Prepare for Quest -> team screen -> Start Quest ->
  (Auto-Battle handles the fight) -> Skip dialogue -> close reward popup
  if shown -> Result screen -> dynamic button back to that category's
  tile grid -> repeat until no NEW tiles remain on any page of that
  category (pages are turned by clicking the numbered page buttons in
  sequence -- 1, 2, 3... -- not the arrow-looking "jump to last page"
  control, which would skip over any NEW content sitting on the pages
  in between; a missing page button means that was the last page) ->
  exit the screen -> Sub Stories grid -> next NEW category -> ... When
  the Sub Stories grid itself shows no NEW badges for a few seconds, the
  bot concludes everything is cleared and stops itself automatically.

  "Exit the screen" can mean two different buttons depending on where the
  bot is: the arrow-style Back button on the category grid, or the
  text-style Close button on a chapter-list screen (e.g. a category's
  full chapter list with CLEAR/NEW rows and page numbers at the bottom).
  The bot tries Back first, then Close, and clicks whichever is present.

SETUP:
1. pip install opencv-python mss pyautogui keyboard numpy pygetwindow
   (tkinter ships with standard Python on Windows -- no separate install needed)
2. Put this file next to a templates/ folder containing:
   skip_button.png, prepare_for_quest.png, start_quest.png, new_badge.png,
   close_reward_popup.png, result_retry.png, result_home.png,
   result_summons.png, next_page_arrow.png, back_button.png, close_button.png,
   page_1.png, page_2.png, page_3.png, page_4.png, page_5.png
3. Run: python subfarm_gui.py
4. Position the game on the Sub Stories screen, then click Start.
   Global hotkeys F4 (start), F11 (pause/resume) and F12 (stop) still work even if
   the GUI window isn't focused, in case you need to bail out fast
   without alt-tabbing.
5. Capture is confined to the game window itself (matched by title, see
   GAME_WINDOW_TITLE below) -- not the whole monitor. Keep this control panel
   from visually overlapping the game window while it's running anyway: capture
   is a screenshot of whatever's on screen in that rectangle, so another window
   dragged on top of the game would still be what gets scanned.
"""

import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import cv2
import numpy as np
import mss
import pyautogui
import keyboard
import pygetwindow as gw

pyautogui.PAUSE = 0  # pyautogui defaults to a 0.1s pause after every call -- that adds up
                      # over thousands of clicks. FAILSAFE stays on (move mouse to a screen
                      # corner to abort) as a manual kill switch.

# Substring match against the window title bar, e.g. "Bleach: Brave Souls" matches
# "Bleach: Brave Souls - BlueStacks" too. Capture is confined to exactly this window's
# rectangle -- nothing outside it (this control panel, the desktop, another app) is ever
# scanned or clicked, no matter what's on screen elsewhere.
GAME_WINDOW_TITLE = "Bleach: Brave Souls"
# How often to re-locate the window and refresh its rectangle, in case it was moved or
# resized while the bot is running. A window handle can go stale (closed/reopened), and
# a title-only lookup can't tell that on its own -- periodic re-checks catch both.
WINDOW_RECHECK_SECONDS = 2.0

# ----------------------------------------------------------------------------
# CONFIG (same defaults as subfarm_bot.py)
# ----------------------------------------------------------------------------

MATCH_CONFIDENCE = 0.82
# NEW_BADGE_CONFIDENCE is on the masked TM_CCORR_NORMED scale (see new_badge_mask.png),
# not the same scale as the other templates' TM_CCOEFF_NORMED matches. Measured: real
# badges score ~0.91-0.95, genuine non-badge screens top out ~0.82-0.83 -- 0.87 sits
# safely in that gap.
NEW_BADGE_CONFIDENCE = 0.87
# The result screen (Retry/Home/Summons row) fades/slides in over ~a second. Waiting for
# a full-confidence match meant waiting for the animation to finish before the bot even
# noticed the screen. These buttons are simple, high-contrast, and always require all
# three to match together anyway, so a lower threshold here is safe and lets the bot
# react as soon as the buttons are legible, not once they're fully settled.
RESULT_SCREEN_CONFIDENCE = 0.68
# next_page_arrow's shape is generic enough that it can weakly false-match unrelated
# icons on other screens (e.g. an enemy portrait on the team/Prepare-for-Quest screen
# matched at 0.843 -- just above the default 0.82). Since a false match here leads to
# the bot concluding "last page reached" and backing out of a screen it shouldn't have
# left, this one needs a tighter bar than the default.
NEXT_PAGE_ARROW_CONFIDENCE = 0.90
# How long after a forward-progress click (NEW badge / Prepare for Quest / Start Quest)
# to suspend the "exhausted, back out" fallback. Each step of entering a quest -- the
# tile click, the Prepare-for-Quest transition, the loading screen, the battle intro --
# produces a blank/unrecognized frame for a moment. Without this grace window, that gap
# gets misread as "nothing left here" and the bot backs out of a quest mid-navigation.
QUEST_LOAD_GRACE_SECONDS = 8.0
# Pagination on category/chapter-list screens uses numbered page buttons (1,2,3,4,5),
# not a "next page" arrow -- the arrow-looking control at the end of that row is
# actually a "jump to last page" button, which would skip straight past pages 2-4 and
# miss any NEW content sitting on them. We click through page_1..page_N in sequence
# instead. Self-match on these digit crops is ~1.0 and the worst cross-digit confusion
# (3 vs 5) is ~0.71, so 0.85 leaves a comfortable margin either way.
PAGE_BUTTON_CONFIDENCE = 0.85
MAX_PAGE_BUTTONS = 5  # highest page number we have a template for
CLICK_JITTER_PX = 3
POST_CLICK_DELAY = 0.12
STUCK_TIMEOUT = 10.0
STUCK_LOG_DIR = "stuck_screens"
PAGE_DIFF_THRESHOLD = 6.0   # mean pixel diff below this = page didn't actually change (last page)
# _frames_differ() for page-turn checks used to diff the ENTIRE captured window -- top bar,
# currency icons, the pagination row itself, borders, etc. A lot of that area is pixel-identical
# from one page to the next, and even within the tile grid every tile shares the same big
# CLEAR/NEW banner graphic, so a real page turn could still average out under the threshold and
# read as "nothing changed." Crop to roughly just the tile-grid band (excludes the header and the
# pagination/bottom-nav strip) before diffing for page-turn purposes specifically, so the actual
# changing content isn't diluted by static chrome. Fractions are of frame height.
CONTENT_REGION_TOP_FRAC = 0.12
CONTENT_REGION_BOTTOM_FRAC = 0.82
# If a page-turn diff still reads as "no change" after the wait windows above, don't immediately
# write off every remaining page -- retry the same click once first. A single missed click
# (jitter landing just off the hitbox, a stray input drop) shouldn't cost pages 3-5 of a category.
MAX_PAGE_TURN_RETRIES = 1
# The page-turn transition animates for a beat after the click. Diffing the very next
# frame (captured just POST_CLICK_DELAY later) against the pre-click frame catches both
# of them mid-animation -- nearly identical to each other regardless of whether the page
# actually changed -- which reads as "no change" and wrongly ends pagination early. Give
# the animation this long before even attempting to judge whether it changed.
PAGE_TURN_SETTLE_SECONDS = 0.4
# Even after the settle window, an occasional slow frame or lag spike could still be
# mid-animation. Rather than judging "no change" as final the instant settle elapses,
# keep giving the animation the benefit of the doubt up to this much longer -- only once
# BOTH windows have elapsed with genuinely no change do we conclude this really is the
# last page. This is also the window the downstream exit-check is gated behind, so it
# can never fire until pagination has actually reached one of those two conclusions.
PAGE_TURN_MAX_WAIT_SECONDS = 1.5
# The NEW ribbon graphics render in AFTER the tile artwork itself -- observed badge counts
# still climbing (9 -> 12) a couple seconds after a page had already visually settled. A
# badge scan that runs the instant a page's layout looks stable can therefore read "zero
# NEW here" before the ribbons have actually finished drawing, and wrongly advance past a
# page that does have NEW content. Once a page is confirmed landed, hold off treating a
# "no badges" read as trustworthy until this much time has passed.
PAGE_LOAD_GRACE_SECONDS = 2.0
MAX_PAGES_SAFETY = 30       # hard safety cap so a matching bug can't infinite-loop pagination
LOOP_DELAY = 0.01           # idle re-scan interval -- lower = faster reaction, more CPU
TEMPLATE_DIR = "templates"

# How many consecutive matching scans are required before acting on a template.
# 1 = act the instant it's seen, with zero debounce. Every template uses this now --
# no more waiting for a second confirming frame on anything.
DEFAULT_CONFIRM_FRAMES = 1
# exit_button (the "nothing left here, back/close out" fallback) gets extra debounce:
# it's already the lowest-priority check in the loop, but with DEFAULT_CONFIRM_FRAMES=1
# a single blank/transitional frame was enough to trigger it. Requiring a few consecutive
# confirming scans costs nothing on a genuinely exhausted screen (that state just sits
# there), but filters out one-off misfires during loading/animation.
CONFIRM_FRAMES_MAP = {"exit_button": 4}

TEMPLATE_NAMES = [
    "skip_button", "prepare_for_quest", "start_quest",
    "new_badge", "close_reward_popup", "tap_screen",
    "result_retry", "result_home", "result_summons",
    "next_page_arrow", "back_button", "close_button",
    "page_1", "page_2", "page_3", "page_4", "page_5",
    "friend_request_cancel", "chapter_uncleared", "chapter_close_button",
]


# ----------------------------------------------------------------------------
# BOT ENGINE (runs on a background thread; talks to the GUI via a queue)
# ----------------------------------------------------------------------------

class SubFarmBot:
    def __init__(self, log_callback):
        self.log = log_callback
        self._templates = {}
        self._masks = {}
        self._thread = None
        self._running = False
        self._paused = False
        self._stop_flag = False
        self._confirm_counts = {}
        self._pages_tried = 0
        self._at_home = True          # True = we believe we're on the Sub Stories category grid
        self._last_page_frame = None  # frame captured right before the last "next page" click
        self._page_click_at = 0.0     # time.time() of the last page-turn click, for settle timing
        self._page_turn_retries = 0   # how many times we've retried the current page-turn click
        self._current_page_settled_at = 0.0  # time.time() when the current page was first
                                              # confirmed landed; 0.0 = not yet confirmed
        self._last_badge_frame = None      # frame captured right before the last NEW-badge click
        self._badge_click_repeats = 0      # consecutive times a badge click produced no screen change
        self._nav_progress_at = 0.0        # time.time() of the last forward-progress click
                                            # (NEW badge / Prepare for Quest / Start Quest)
        self.on_finished = None       # optional callback(reason: str) fired when farming completes

    # -- lifecycle -----------------------------------------------------
    def start(self):
        if self._running:
            return
        try:
            self._load_templates()
        except FileNotFoundError as e:
            self.log(f"ERROR: {e}")
            self.log("Check that templates/ is next to this script and has all 10 PNGs.")
            return

        self._stop_flag = False
        self._paused = False
        self._running = True
        self._pages_tried = 0
        self._at_home = True
        self._last_page_frame = None
        self._page_click_at = 0.0
        self._page_turn_retries = 0
        self._current_page_settled_at = 0.0
        self._last_badge_frame = None
        self._badge_click_repeats = 0
        self._nav_progress_at = 0.0
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.log("Bot started. Assuming current screen is Sub Stories (category grid).")

    def stop(self):
        self._stop_flag = True
        self._running = False
        self.log("Stopping...")

    def toggle_pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        self.log("Paused." if self._paused else "Resumed.")

    @property
    def running(self):
        return self._running

    @property
    def paused(self):
        return self._paused

    # -- template loading / matching ------------------------------------
    def _load_templates(self):
        for name in TEMPLATE_NAMES:
            path = os.path.join(TEMPLATE_DIR, f"{name}.png")
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Missing template: {path}")
            # Matching is done in grayscale: matchTemplate on 1 channel instead of 3 is
            # roughly 3x cheaper, and every one of these UI elements has enough contrast
            # that dropping color costs essentially no accuracy.
            self._templates[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Optional mask (e.g. new_badge_mask.png) marks which pixels are reliably
            # part of the icon itself vs. background/artwork that varies by context.
            # When present, matching uses TM_CCORR_NORMED with this mask instead of
            # plain TM_CCOEFF_NORMED, which is far more robust to varying backgrounds.
            mask_path = os.path.join(TEMPLATE_DIR, f"{name}_mask.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self._masks[name] = mask  # already single-channel -- matches the gray frame directly

    # A normalized correlation score can never legitimately exceed 1.0 (a perfect
    # match scores exactly 1.0). NaN/Inf is the extreme case of a near-zero-variance
    # region breaking the match math; a finite-but-impossible value like 1.16 is the
    # same failure mode at a milder magnitude, and just as capable of falsely beating
    # a confidence threshold. Anything above this is discarded rather than trusted --
    # a small allowance above 1.0 for ordinary floating-point rounding.
    MAX_PLAUSIBLE_SCORE = 1.02

    def _sanitize_match_result(self, result):
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        result[result > self.MAX_PLAUSIBLE_SCORE] = -1.0
        return result

    def _match_single(self, frame_gray, name, confidence=MATCH_CONFIDENCE):
        tmpl = self._templates[name]
        mask = self._masks.get(name)
        if mask is not None:
            result = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            result = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        # A masked TM_CCORR_NORMED match divides by the local image variance under the
        # mask. Over a flat/low-variance region (e.g. a solid-color panel with no game
        # content in it) that denominator collapses toward zero and the result can come
        # back as NaN/Inf, or just an impossibly large finite value -- either way it
        # would trivially beat any real confidence threshold. Clamp those out before
        # they can ever be read as a match.
        result = self._sanitize_match_result(result)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < confidence:
            return None
        h, w = tmpl.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2, max_val)

    def _match_all(self, frame_gray, name, confidence=NEW_BADGE_CONFIDENCE, min_dist=40):
        tmpl = self._templates[name]
        mask = self._masks.get(name)
        h, w = tmpl.shape[:2]
        if mask is not None:
            result = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            result = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        # Same clamp as _match_single -- see comment there. Without this, a flat
        # region can produce hundreds of spurious "matches" that all beat the
        # confidence threshold simultaneously.
        result = self._sanitize_match_result(result)
        ys, xs = np.where(result >= confidence)
        candidates = sorted(
            ((xs[i], ys[i], result[ys[i], xs[i]]) for i in range(len(xs))),
            key=lambda c: -c[2],
        )
        kept = []
        for x, y, conf in candidates:
            cx, cy = x + w // 2, y + h // 2
            if all(abs(cx - kx) > min_dist or abs(cy - ky) > min_dist for kx, ky, _ in kept):
                kept.append((cx, cy, conf))
        kept.sort(key=lambda c: (c[1] // 60, c[0]))
        return kept

    def _confirm(self, name, matched):
        if not matched:
            self._confirm_counts[name] = 0
            return False
        self._confirm_counts[name] = self._confirm_counts.get(name, 0) + 1
        required = CONFIRM_FRAMES_MAP.get(name, DEFAULT_CONFIRM_FRAMES)
        return self._confirm_counts[name] >= required

    def _reset_confirm(self, name):
        self._confirm_counts[name] = 0

    def _click(self, monitor, x, y):
        jx = np.random.randint(-CLICK_JITTER_PX, CLICK_JITTER_PX + 1)
        jy = np.random.randint(-CLICK_JITTER_PX, CLICK_JITTER_PX + 1)
        pyautogui.click(monitor["left"] + x + jx, monitor["top"] + y + jy)

    def _match_exit(self, frame_gray, back_button_match=None):
        """Some screens exit via the arrow-style 'Back' button (category grid),
        others via a text-style 'Close' button (e.g. a chapter list screen).
        Close is checked first: when a chapter-list modal is open, the category
        grid's Back arrow is still visible (just dimmed) underneath it, and
        template matching is tolerant enough of that dimming to still match --
        so checking Back first would win incorrectly and never actually close
        the modal. Checking Close first ensures an open modal gets closed
        before we ever consider the dimmed Back arrow behind it.

        back_button_match is an optional (x, y, conf) already computed by the
        caller this tick (same confidence as our own default) -- passing it in
        avoids matching the same "back_button" template against the same frame
        more than once per loop tick, since this gets called from more than one
        place (badge-exhausted fallback, the general exit-check) and the caller
        also needs a back_button read of its own right after."""
        m = self._match_single(frame_gray, "close_button")
        if m:
            x, y, conf = m
            return (x, y, conf, "close_button")
        m = back_button_match if back_button_match is not None else \
            self._match_single(frame_gray, "back_button")
        if m:
            x, y, conf = m
            return (x, y, conf, "back_button")
        return None

    def _frames_differ(self, a, b):
        """True if two frames look meaningfully different (i.e. the page actually turned)."""
        if a is None or b is None or a.shape != b.shape:
            return True
        return float(np.mean(cv2.absdiff(a, b))) >= PAGE_DIFF_THRESHOLD

    @staticmethod
    def _crop_content_region(frame):
        """Crop out the static header and the pagination/bottom-nav strip, leaving roughly
        just the tile-grid band. Used for page-turn diffs so a real content change isn't
        diluted (and hidden below threshold) by all the pixel-identical chrome around it."""
        h = frame.shape[0]
        top = int(h * CONTENT_REGION_TOP_FRAC)
        bottom = int(h * CONTENT_REGION_BOTTOM_FRAC)
        return frame[top:bottom, :]

    def _frames_differ_content(self, a, b):
        """Same as _frames_differ, but restricted to the tile-grid region -- see
        _crop_content_region for why the page-turn check specifically needs this."""
        if a is None or b is None or a.shape != b.shape:
            return True
        return self._frames_differ(self._crop_content_region(a), self._crop_content_region(b))

    def _find_game_region(self):
        """Locate the real game window by title and return an mss-style capture
        region {left, top, width, height} confined to exactly that window. Returns
        None if no matching, non-minimized window is currently found -- callers
        must treat that as "can't safely proceed," never fall back to grabbing the
        whole screen. Grabbing the whole screen is what let the bot scan (and
        click into) its own control panel in the first place."""
        try:
            wins = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
        except Exception as e:
            self.log(f"ERROR: window lookup failed ({e}).")
            return None
        wins = [
            w for w in wins
            if w.title and GAME_WINDOW_TITLE.lower() in w.title.lower()
            and w.width > 0 and w.height > 0 and not w.isMinimized
        ]
        if not wins:
            return None
        win = wins[0]
        return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}

    # -- main loop -------------------------------------------------------
    def _run_loop(self):
        os.makedirs(STUCK_LOG_DIR, exist_ok=True)
        last_activity = time.time()
        stuck_shot_taken = False
        last_window_check = 0.0
        monitor = None

        with mss.mss() as sct:
            while not self._stop_flag:
                if self._paused:
                    time.sleep(0.2)
                    continue

                now = time.time()
                if monitor is None or (now - last_window_check) >= WINDOW_RECHECK_SECONDS:
                    found = self._find_game_region()
                    last_window_check = now
                    if found is None:
                        self.log(f"ERROR: no window matching '{GAME_WINDOW_TITLE}' found "
                                 f"(or it's minimized). Stopping rather than risk scanning "
                                 f"something else -- reopen/restore the game and hit Start again.")
                        self._running = False
                        return
                    monitor = found

                frame = np.array(sct.grab(monitor))[:, :, :3]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                now = time.time()
                acted = False

                # "back_button" gets checked from up to three places in a single tick
                # (badge-exhausted fallback, the general exit-check, and the final
                # "are we really home" check) -- memoize it per-frame so at most one of
                # those actually pays for the template match; the other(s) reuse it.
                _back_button_cache = "UNSET"
                def get_back_button_match():
                    nonlocal _back_button_cache
                    if _back_button_cache == "UNSET":
                        _back_button_cache = self._match_single(frame_gray, "back_button")
                    return _back_button_cache

                m = self._match_single(frame_gray, "friend_request_cancel")
                if self._confirm("friend_request_cancel", m is not None):
                    x, y, conf = m
                    self.log(f"Friend request popup -> clicking Cancel (conf {conf:.2f})")
                    self._click(monitor, x, y)
                    self._reset_confirm("friend_request_cancel")
                    acted = True

                if not acted:
                    m = self._match_single(frame_gray, "skip_button")
                    if self._confirm("skip_button", m is not None):
                        x, y, conf = m
                        self.log(f"Skip dialogue (conf {conf:.2f})")
                        self._click(monitor, x, y)
                        self._reset_confirm("skip_button")
                        acted = True

                if not acted:
                    retry = self._match_single(frame_gray, "result_retry", confidence=RESULT_SCREEN_CONFIDENCE)
                    home = self._match_single(frame_gray, "result_home", confidence=RESULT_SCREEN_CONFIDENCE)
                    summons = self._match_single(frame_gray, "result_summons", confidence=RESULT_SCREEN_CONFIDENCE)
                    on_result = retry and home and summons
                    if self._confirm("result_screen", on_result):
                        spacing = home[0] - retry[0]
                        tx, ty = summons[0] + spacing, summons[1]
                        self.log(f"Result screen -> back to category grid ({tx},{ty})")
                        self._click(monitor, tx, ty)
                        self._reset_confirm("result_screen")
                        # Dynamic button always returns to that category's tile grid,
                        # never all the way to Sub Stories -- so we're NOT home yet.
                        self._at_home = False
                        self._pages_tried = 0
                        self._page_turn_retries = 0
                        self._current_page_settled_at = 0.0
                        self._last_page_frame = None
                        # This return trip has its own loading blip, same as entering a
                        # quest does. Every other forward-progress click refreshes the
                        # grace period; this one didn't, so by the time we land back on
                        # the grid the grace window (usually started minutes ago, before
                        # the battle) has already expired -- leaving only the 4-frame
                        # debounce to protect against misreading the landing frame as
                        # "nothing left here, back out."
                        self._nav_progress_at = now
                        acted = True

                if not acted:
                    m = self._match_single(frame_gray, "close_reward_popup")
                    if self._confirm("close_reward_popup", m is not None):
                        x, y, conf = m
                        self.log(f"Closing reward popup (conf {conf:.2f})")
                        self._click(monitor, x, y)
                        self._reset_confirm("close_reward_popup")
                        acted = True

                if not acted:
                    m = self._match_single(frame_gray, "tap_screen")
                    if self._confirm("tap_screen", m is not None):
                        x, y, conf = m
                        self.log(f"EXP screen -> tap to continue (conf {conf:.2f})")
                        self._click(monitor, x, y)
                        self._reset_confirm("tap_screen")
                        acted = True

                if not acted:
                    m = self._match_single(frame_gray, "prepare_for_quest")
                    if self._confirm("prepare_for_quest", m is not None):
                        uncleared = self._match_single(frame_gray, "chapter_uncleared") is not None
                        if uncleared:
                            x, y, conf = m
                            self.log(f"Prepare for Quest (conf {conf:.2f})")
                            self._click(monitor, x, y)
                            self._nav_progress_at = now
                        else:
                            cm = self._match_single(frame_gray, "chapter_close_button")
                            if cm:
                                x, y, conf = cm
                                self.log(f"Chapter already cleared (Best Time set, not "
                                         f"'Uncleared') -> clicking this screen's Close "
                                         f"(conf {conf:.2f})")
                                self._click(monitor, x, y)
                                self._nav_progress_at = now
                            else:
                                self.log("Chapter already cleared, but this screen's Close "
                                         "button wasn't found -- staying put")
                        self._reset_confirm("prepare_for_quest")
                        acted = True

                if not acted:
                    m = self._match_single(frame_gray, "start_quest")
                    if self._confirm("start_quest", m is not None):
                        x, y, conf = m
                        self.log(f"Start Quest (conf {conf:.2f})")
                        self._click(monitor, x, y)
                        self._reset_confirm("start_quest")
                        self._nav_progress_at = now
                        acted = True

                if not acted:
                    badges = self._match_all(frame_gray, "new_badge")
                    if self._confirm("new_badge", len(badges) > 0):
                        if self._last_badge_frame is not None and \
                                not self._frames_differ(frame, self._last_badge_frame):
                            # Same NEW badge(s), same screen, no change since the last click --
                            # clicking isn't actually going anywhere (permanent/undismissable
                            # NEW ribbon, or a click that isn't registering). Don't loop on it
                            # forever; treat this screen as exhausted and back/close out.
                            self._badge_click_repeats += 1
                        else:
                            self._badge_click_repeats = 0

                        if self._badge_click_repeats >= 2:
                            mb = self._match_exit(frame_gray, get_back_button_match())
                            if mb:
                                x, y, _, which = mb
                                self.log(f"NEW badge click had no effect twice in a row -> "
                                         f"treating as exhausted, clicking {which}")
                                self._click(monitor, x, y)
                                # close_button only closes a chapter-list modal back to that
                                # category's story-tile grid -- still not home. Only
                                # back_button actually returns to the real Sub Stories grid.
                                self._at_home = (which == "back_button")
                            else:
                                self.log("NEW badge click had no effect twice in a row, but "
                                         "no Back/Close control found -- staying put")
                            self._reset_confirm("new_badge")
                            self._badge_click_repeats = 0
                            self._last_badge_frame = None
                            self._pages_tried = 0
                            self._page_turn_retries = 0
                            self._current_page_settled_at = 0.0
                            self._last_page_frame = None
                            acted = True
                        else:
                            x, y, conf = badges[0]
                            self.log(f"Found {len(badges)} NEW badge(s) -> clicking first "
                                     f"(conf {conf:.2f})")
                            self._last_badge_frame = frame.copy()
                            self._click(monitor, x, y)
                            self._reset_confirm("new_badge")
                            self._nav_progress_at = now
                            # Whether this was a category tile (leaving Sub Stories) or a story
                            # tile (already inside a category), we're no longer on the home grid.
                            self._at_home = False
                            self._pages_tried = 0
                            self._page_turn_retries = 0
                            self._current_page_settled_at = 0.0
                            self._last_page_frame = None
                            acted = True

                if not acted:
                    # self._pages_tried counts how many page-advance clicks we've made
                    # since arriving at page 1, so we're currently on page (pages_tried+1)
                    # and want to click the button for page (pages_tried+2) next.
                    current_page = self._pages_tried + 1
                    target_page = self._pages_tried + 2
                    if target_page <= MAX_PAGE_BUTTONS:
                        m = self._match_single(frame_gray, f"page_{target_page}",
                                                confidence=PAGE_BUTTON_CONFIDENCE)
                        if self._confirm("page_button", m is not None):
                            elapsed_since_click = now - self._page_click_at
                            still_settling = elapsed_since_click < PAGE_TURN_SETTLE_SECONDS
                            past_max_wait = elapsed_since_click >= PAGE_TURN_MAX_WAIT_SECONDS
                            if self._last_page_frame is not None and still_settling:
                                # The page-turn animation from the previous click hasn't had
                                # time to finish yet -- this frame and the pre-click frame we
                                # saved are both still mid-transition, so diffing them now
                                # would compare "mid-transition" against itself and wrongly
                                # read as "nothing changed." Wait for it to settle instead of
                                # judging on this frame.
                                pass
                            elif self._last_page_frame is not None and \
                                    not self._frames_differ_content(frame, self._last_page_frame) and \
                                    not past_max_wait:
                                # Past the initial settle window but still no visible change --
                                # could just be an unusually slow animation (lag spike, slower
                                # device). Keep giving it the benefit of the doubt up to the
                                # longer max-wait ceiling instead of concluding "last page" the
                                # instant the short settle window expires.
                                pass
                            elif self._last_page_frame is not None and \
                                    not self._frames_differ_content(frame, self._last_page_frame):
                                # The click that was supposed to take us from (current_page - 1)
                                # to current_page doesn't look like it changed anything. That's
                                # sometimes a real "no more pages," but it can also be a false
                                # negative (missed hitbox, a slow-loading page) -- one inconclusive
                                # read shouldn't cost every remaining page in this category. Retry
                                # the same click a limited number of times before giving up.
                                if self._page_turn_retries < MAX_PAGE_TURN_RETRIES:
                                    retry_m = self._match_single(
                                        frame_gray, f"page_{current_page}",
                                        confidence=PAGE_BUTTON_CONFIDENCE)
                                    if retry_m:
                                        rx, ry, rconf = retry_m
                                        self._page_turn_retries += 1
                                        self.log(f"Page {current_page} click had no visible "
                                                 f"effect -> retrying "
                                                 f"({self._page_turn_retries}/{MAX_PAGE_TURN_RETRIES})")
                                        self._last_page_frame = frame.copy()
                                        self._page_click_at = now
                                        self._click(monitor, rx, ry)
                                        self._reset_confirm("page_button")
                                        acted = True
                                    else:
                                        # Can't even find the button to retry anymore -- treat
                                        # this as the last page rather than getting stuck here.
                                        self.log(f"Page {current_page} click had no visible "
                                                 f"effect and its button is no longer visible "
                                                 f"-> treating current page as the last one")
                                        self._pages_tried = MAX_PAGE_BUTTONS
                                        self._last_page_frame = None
                                        self._page_turn_retries = 0
                                        self._current_page_settled_at = 0.0
                                        self._reset_confirm("page_button")
                                else:
                                    # Retry(ies) exhausted too -- push it past MAX_PAGE_BUTTONS so
                                    # the exhausted/back-out check below handles it on this same
                                    # frame, instead of racing through the remaining numbered
                                    # buttons with nothing behind them.
                                    self.log(f"Page {current_page} click had no visible effect "
                                             f"after retry -> treating current page as the last one")
                                    self._pages_tried = MAX_PAGE_BUTTONS
                                    self._last_page_frame = None
                                    self._page_turn_retries = 0
                                    self._current_page_settled_at = 0.0
                                    self._reset_confirm("page_button")
                            else:
                                # Page-turn confirmed (or, for the very first page, assumed):
                                # we're genuinely sitting on current_page with zero NEW badges
                                # seen this tick. That doesn't mean there really are none --
                                # ribbons can still be rendering in, so don't trust "empty"
                                # until this page has had a fair chunk of time to finish
                                # drawing. Badge scanning upstream still runs every tick
                                # regardless, so a badge that appears mid-wait gets caught
                                # immediately; this only gates the "give up and move on" step.
                                if self._current_page_settled_at == 0.0:
                                    self._current_page_settled_at = now
                                if (now - self._current_page_settled_at) < PAGE_LOAD_GRACE_SECONDS:
                                    pass
                                else:
                                    x, y, conf = m
                                    self._pages_tried += 1
                                    self._page_turn_retries = 0
                                    self._current_page_settled_at = 0.0
                                    self.log(f"No NEW here -> clicking page {target_page} (conf {conf:.2f})")
                                    self._last_page_frame = frame.copy()
                                    self._page_click_at = now
                                    self._click(monitor, x, y)
                                    self._reset_confirm("page_button")
                                    acted = True

                in_nav_grace_period = (now - self._nav_progress_at) < QUEST_LOAD_GRACE_SECONDS
                # A page-turn click is still awaiting its settle/max-wait window (see
                # PAGE_TURN_SETTLE_SECONDS / PAGE_TURN_MAX_WAIT_SECONDS above) -- the
                # page-turn block deliberately leaves `acted` False while it waits, but
                # that doesn't mean this screen is exhausted. Gate on the full max-wait
                # window (not just the short settle window) so the exit-check truly can't
                # fire until the pagination block has reached one of its two real
                # conclusions (page changed -> keep going, or genuinely didn't -> last
                # page) -- never while it's still mid-animation, no matter how slow that
                # animation runs.
                page_turn_pending = self._last_page_frame is not None and \
                    (now - self._page_click_at) < PAGE_TURN_MAX_WAIT_SECONDS

                # On the last page (target_page > MAX_PAGE_BUTTONS) the page-button block above
                # never runs at all, so _current_page_settled_at never gets set there -- this is
                # the only grace this page gets against the same late-rendering-ribbons problem.
                # _page_click_at still reflects the click that brought us to this page, so reuse
                # it the same way.
                page_ribbon_load_pending = self._last_page_frame is not None and \
                    (now - self._page_click_at) < PAGE_LOAD_GRACE_SECONDS

                # We used to also require a "page_1" match here as positive confirmation
                # that this is a genuine category/chapter-list screen before ever trying
                # to exit it -- meant to rule out mid-battle/loading frames that simply
                # didn't match anything. But page_1.png was captured showing the
                # UNselected look of the "1" button; the instant the bot actually lands
                # on page 1 of a fresh category (i.e. every time), "1" renders in its
                # active/selected style instead and never matches -- which silently
                # blocked this whole branch forever on exactly the screens it exists to
                # handle. match_exit() finding a real Close/Back button is already
                # positive, specific evidence this is an exit-able screen; the
                # CONFIRM_FRAMES_MAP debounce below (4 consecutive frames) is what
                # filters out one-off transitional mismatches, so the extra page_1 gate
                # was redundant on top of being broken.
                if not acted and not self._at_home and not in_nav_grace_period \
                        and not page_turn_pending and not page_ribbon_load_pending:
                    # No NEW badges, and no further page to click -- this category/chapter-list
                    # is genuinely exhausted. Exit via whichever control this screen actually has
                    # (arrow-style Back on the category grid, or a text-style Close button on a
                    # chapter list).
                    m = self._match_exit(frame_gray, get_back_button_match())
                    if self._confirm("exit_button", m is not None):
                        x, y, conf, which = m
                        self.log(f"No NEW and no pagination here -> exhausted, clicking {which}")
                        self._click(monitor, x, y)
                        self._reset_confirm("exit_button")
                        self._pages_tried = 0
                        self._page_turn_retries = 0
                        self._current_page_settled_at = 0.0
                        self._last_page_frame = None
                        # Same distinction as above: close_button only backs out of a
                        # chapter-list modal to the story-tile grid, not all the way home.
                        self._at_home = (which == "back_button")
                        acted = True

                if acted:
                    last_activity = now
                    stuck_shot_taken = False
                    time.sleep(POST_CLICK_DELAY)
                else:
                    # Positive confirmation that we're actually looking at a screen with
                    # a Back control on it -- not just trusting the _at_home flag, which
                    # is set on a few unverified assumptions (bot start(), or "exit_button"
                    # having been clicked) and could be wrong, e.g. if the game wasn't
                    # actually sitting on the Sub Stories grid when Start was pressed.
                    # Declaring the whole run "complete" is a big, final claim -- it
                    # should require real evidence, not just "nothing matched and we
                    # assumed we were home."
                    on_home_screen = get_back_button_match() is not None

                    if self._at_home and on_home_screen:
                        if not stuck_shot_taken and (now - last_activity) >= STUCK_TIMEOUT:
                            fname = os.path.join(STUCK_LOG_DIR, f"stuck_{int(now)}.png")
                            cv2.imwrite(fname, frame)
                            self.log(f"No NEW badges anywhere on Sub Stories for "
                                     f"{STUCK_TIMEOUT:.0f}s -- all stories cleared. Stopping.")
                            stuck_shot_taken = True
                            self._stop_flag = True
                            if self.on_finished:
                                try:
                                    self.on_finished("complete")
                                except Exception:
                                    pass
                    else:
                        if not stuck_shot_taken and (now - last_activity) >= STUCK_TIMEOUT:
                            fname = os.path.join(STUCK_LOG_DIR, f"stuck_{int(now)}.png")
                            cv2.imwrite(fname, frame)
                            self.log(f"No recognized screen for {STUCK_TIMEOUT:.0f}s -- "
                                     f"saved {fname}. Bot may be stuck on an unhandled screen.")
                            stuck_shot_taken = True
                    time.sleep(LOOP_DELAY)

        self._running = False
        self.log("Stopped.")


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("BBS Sub-Story Auto-Farmer")
        root.geometry("560x420")
        root.resizable(False, False)

        self._log_queue = queue.Queue()
        self.bot = SubFarmBot(log_callback=self._enqueue_log)

        # -- top bar: status + controls --
        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        self.status_var = tk.StringVar(value="Stopped")
        status_label = ttk.Label(top, textvariable=self.status_var,
                                  font=("Segoe UI", 14, "bold"))
        status_label.pack(side="left")

        btn_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        btn_frame.pack(fill="x")

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 5))

        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self.on_pause,
                                     state="disabled")
        self.pause_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.on_stop,
                                    state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        hint = ttk.Label(root, text="Global hotkeys still work: F4 start/stop, F11 pause/resume",
                          foreground="#666", padding=(10, 0))
        hint.pack(fill="x")

        # -- log window --
        log_frame = ttk.Frame(root, padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_box = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled",
                                                   font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)

        # global hotkeys work regardless of window focus
        keyboard.add_hotkey("f4", self.on_toggle_start_stop)
        keyboard.add_hotkey("f11", self.on_pause)
        keyboard.add_hotkey("f12", self.on_stop)  # kept as a backup stop-only key

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log_queue()

    # -- button handlers --------------------------------------------------
    def on_toggle_start_stop(self):
        if self.bot.running:
            self.on_stop()
        else:
            self.on_start()

    def on_start(self):
        if self.bot.running:
            return
        self.bot.start()
        self.status_var.set("Running")
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="Pause")
        self.stop_btn.config(state="normal")

    def on_pause(self):
        if not self.bot.running:
            return
        self.bot.toggle_pause()
        if self.bot.paused:
            self.status_var.set("Paused")
            self.pause_btn.config(text="Resume")
        else:
            self.status_var.set("Running")
            self.pause_btn.config(text="Pause")

    def on_stop(self):
        if not self.bot.running:
            return
        self.bot.stop()
        self._reset_controls_to_stopped()

    def _reset_controls_to_stopped(self):
        self.status_var.set("Stopped")
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")

    def _on_close(self):
        self.bot.stop()
        self.root.after(200, self.root.destroy)

    # -- log plumbing (thread -> GUI) --------------------------------------
    def _enqueue_log(self, message: str):
        self._log_queue.put(message)

    def _poll_log_queue(self):
        try:
            while True:
                message = self._log_queue.get_nowait()
                timestamp = time.strftime("%H:%M:%S")
                self.log_box.config(state="normal")
                self.log_box.insert("end", f"[{timestamp}] {message}\n")
                self.log_box.see("end")
                self.log_box.config(state="disabled")
        except queue.Empty:
            pass
        # The bot can stop itself (e.g. it decided all Sub Stories are cleared).
        # If that happened, the buttons are still showing "Running" -- fix that.
        if not self.bot.running and self.stop_btn["state"] == "normal":
            self._reset_controls_to_stopped()
        self.root.after(150, self._poll_log_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop() 
import dataclasses
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rep_counter import SQUAT, RepCounter
from squat_assess import (CAT_NA, CAT_PARTIAL, CAT_REP, ERR, NA, NORM, WARN, AttemptResult,
                          Calibrator, SquatAssessor, Thresholds, angle3, rating_from_mean,
                          score_from_rules, summarize, thigh_tilt, trunk_lean)
import synth

FPS = 30.0


def run(frames, th=None):
    counter = RepCounter(dataclasses.replace(SQUAT, fps=FPS))
    asr = SquatAssessor(FPS, th)
    out = []
    for k in frames:
        sample, ev = counter.update(k)
        out += asr.update(sample.frame_idx, k, ev, counter.state)
    out += asr.flush(len(frames) - 1)
    return counter, asr, out


class Geometry(unittest.TestCase):
    def test_synthetic_pose_angles(self):
        k = synth.pose(100.0, 30.0)
        self.assertAlmostEqual(float(angle3(k[11, :2], k[13, :2], k[15, :2])), 100.0, places=3)
        self.assertAlmostEqual(float(trunk_lean(k[11, :2], k[5, :2])), 30.0, places=3)

    def test_thigh_tilt(self):
        self.assertAlmostEqual(float(thigh_tilt(np.array([0.0, 0.0]), np.array([100.0, 0.0]))), 0.0)
        self.assertAlmostEqual(float(thigh_tilt(np.array([0.0, 0.0]), np.array([0.0, 100.0]))), 90.0)
        self.assertLess(float(thigh_tilt(np.array([0.0, 100.0]), np.array([100.0, 0.0]))), 0)


class Scoring(unittest.TestCase):
    def setUp(self):
        self.th = Thresholds()

    def test_score(self):
        self.assertEqual(score_from_rules({"E1": NORM, "E2": NORM}, self.th), 100)
        self.assertEqual(score_from_rules({"E1": WARN, "E2": NORM}, self.th), 90)
        self.assertEqual(score_from_rules({"E1": ERR, "E2": WARN, "E3": NA}, self.th), 65)

    def test_rating_bounds(self):
        for mean, r in [(100, 5), (90, 5), (89.9, 4), (75, 4), (74, 3), (60, 3), (59, 2), (40, 2), (39, 1), (0, 1)]:
            self.assertEqual(rating_from_mean(mean), r, mean)

    def test_summary_and_preliminary(self):
        def mk(cat, score):
            return AttemptResult(1, cat, 0, 1, 2, score=score)
        s = summarize([mk(CAT_REP, 100), mk(CAT_REP, 90), mk(CAT_PARTIAL, 75), mk(CAT_NA, None)])
        self.assertEqual((s.reps, s.partial, s.not_evaluated), (2, 1, 1))
        self.assertAlmostEqual(s.mean_score, (100 + 90 + 75) / 3)
        self.assertFalse(s.preliminary)
        s = summarize([mk(CAT_REP, 100), mk(CAT_NA, None)])
        self.assertTrue(s.preliminary)  # не оценена половина


class Calibration(unittest.TestCase):
    def test_calibration_success(self):
        c = Calibrator(FPS)
        res = None
        for _ in range(100):
            res = c.update(synth.pose(170, 2)) or res
        self.assertIsNotNone(res)
        self.assertTrue(res.view_ok)
        self.assertAlmostEqual(res.knee_top, 170, delta=0.5)
        self.assertAlmostEqual(res.ratio, synth.THIGH / synth.TORSO, delta=0.01)

    def test_motion_restarts(self):
        c = Calibrator(FPS)
        for i in range(200):
            k = synth.pose(170 - 30 * (i % 2), 2)  # дёргается
            self.assertIsNone(c.update(k))

    def test_bad_view(self):
        c = Calibrator(FPS)
        res = None
        for _ in range(100):
            res = c.update(synth.pose(170, 2, view=0.4)) or res
        self.assertFalse(res.view_ok)


class Attempts(unittest.TestCase):
    def one(self, **kw):
        counter, asr, out = run(synth.squat_frames(**kw))
        self.assertEqual(len(out), 1, [r.summary_line() for r in out])
        return counter, out[0]

    def test_correct_squat(self):
        counter, r = self.one()
        self.assertEqual(counter.count, 1)
        self.assertEqual(r.category, CAT_REP)
        self.assertEqual(r.rules, {"E1": NORM, "E2": NORM, "E3": NORM, "E4": NORM})
        self.assertEqual(r.score, 100)

    def test_e1_shallow(self):
        _, r = self.one(bottom=95.0, lean_bottom=14.0)
        self.assertEqual(r.category, CAT_REP)
        self.assertEqual(r.rules["E1"], ERR)

    def test_partial_attempt_not_counted(self):
        counter, r = self.one(top=165.0, bottom=137.0, lean_bottom=5.0)  # размах 28°
        self.assertEqual(counter.count, 0)
        self.assertEqual(r.category, CAT_PARTIAL)
        self.assertEqual(r.rules["E1"], ERR)

    def test_nuisance_ignored(self):
        counter, asr, out = run(synth.squat_frames(bottom=160.0, lean_bottom=3.0))  # размах 10
        self.assertEqual(counter.count, 0)
        self.assertEqual(out, [])

    def test_knee_cycle_with_stationary_hips_is_ignored(self):
        for bottom in (137.0, 120.0):
            with self.subTest(bottom=bottom):
                frames = [pose.copy() for pose in synth.squat_frames(top=165.0, bottom=bottom)]
                standing_hip_y = frames[0][11, 1]
                for pose in frames:
                    # Сохраняем углы суставов, удерживая таз на одной высоте.
                    pose[:, 1] += standing_hip_y - pose[11, 1]
                counter, _, out = run(frames)
                self.assertEqual(counter.count, 0)
                self.assertEqual(out, [])

    def test_e2_lean(self):
        _, r = self.one(lean_bottom=55.0)
        self.assertEqual(r.rules["E2"], ERR)
        self.assertEqual(r.rules["E1"], NORM)

    def test_e3_hip_ahead(self):
        _, r = self.one(lean_bottom=40.0, hip_ahead=1.0)
        self.assertNotEqual(r.rules["E3"], NORM)
        _, ok = self.one(lean_bottom=40.0, hip_ahead=0.0)
        self.assertEqual(ok.rules["E3"], NORM)

    def test_e3_lean_growth_feature(self):
        _, r = self.one(lean_bottom=40.0, hip_ahead=1.0)
        self.assertGreater(r.features["F9"], 2.0)
        _, ok = self.one(lean_bottom=40.0, hip_ahead=0.0)
        self.assertLess(ok.features["F9"], 2.0)

    def test_deep_squat_with_small_counter_amplitude_not_partial(self):
        # счётчик не засчитал (неверный верх), но глубина достаточна -> не оценено
        from squat_assess import assess_attempt
        frames = synth.squat_frames()
        counter = RepCounter(dataclasses.replace(SQUAT, fps=FPS))
        asr = SquatAssessor(FPS)
        ev = None
        for k in frames:
            s, e = counter.update(k)
            asr.update(s.frame_idx, k, e, counter.state)
            ev = e or ev
        bad = dataclasses.replace(ev, valid=False, reason="мала амплитуда")
        r = assess_attempt(asr.buf, bad, len(frames) - 1, asr.calib, asr.th, FPS, 1)
        self.assertEqual(r.category, CAT_NA)

    def test_camera_roll_is_corrected(self):
        # кадр повёрнут на 6 град.: признаки должны почти не измениться
        frames = synth.squat_frames(bottom=55.0, lean_bottom=22.0)
        a = np.radians(6.0)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        rot = []
        for k in frames:
            k2 = k.copy()
            k2[:, :2] = k[:, :2] @ R.T
            rot.append(k2)
        _, r0 = self.one()
        _, asr, out = run(rot)
        r1 = out[0]
        self.assertAlmostEqual(asr.calib.roll, 6.0, delta=3.0)
        self.assertAlmostEqual(r1.features["F2"], r0.features["F2"], delta=2.0)
        self.assertAlmostEqual(r1.features["F3"], r0.features["F3"], delta=2.0)
        self.assertEqual(r1.rules, r0.rules)

    def test_e4_no_full_extension(self):
        _, r = self.one(top_after=145.0)
        self.assertEqual(r.rules["E4"], ERR)

    def test_low_confidence_not_evaluated(self):
        _, r = self.one(low_conf_bottom=True, hold_s=1.5)
        self.assertEqual(r.category, CAT_NA)
        self.assertIsNone(r.score)

    def test_too_fast_not_evaluated(self):
        counter, r = self.one(down_s=0.2, up_s=0.2, hold_s=0.0)
        self.assertEqual(counter.count, 0)
        self.assertEqual(r.category, CAT_NA)
        self.assertEqual(r.note, "слишком быстро")

    def test_timer_reset_not_evaluated(self):
        counter, r = self.one(hold_s=12.0)
        self.assertEqual(r.category, CAT_NA)
        self.assertEqual(r.note, "сброс по таймеру")

    def test_bad_view_not_evaluated(self):
        _, r = self.one(view=0.4)
        self.assertEqual(r.category, CAT_NA)

    def test_series_of_five(self):
        counter, asr, out = run(synth.squat_frames(n_reps=5))
        self.assertEqual(counter.count, 5)
        self.assertEqual([r.category for r in out], [CAT_REP] * 5)
        s = asr.summary()
        self.assertEqual((s.reps, s.rating, s.preliminary), (5, 5, False))

    def test_counter_unchanged_by_assessor(self):
        frames = synth.squat_frames(n_reps=3)
        c1, _, _ = run(frames)
        c2 = RepCounter(dataclasses.replace(SQUAT, fps=FPS))
        for k in frames:
            c2.update(k)
        self.assertEqual(c1.count, c2.count)


if __name__ == "__main__":
    unittest.main()

"use strict";

const API_VERSION = "assessment-2";
const ruleLabels = {E1: "Глубина", E2: "Наклон", E3: "Подъём таза", E4: "Выпрямление"};
const ruleStyles = {"норма": "ok", "предупреждение": "warning", "ошибка": "error"};

const $ = (id) => document.getElementById(id);
const visible = (id, show) => $(id).classList.toggle("hidden", !show);
const camera = $("camera");
const capture = document.createElement("canvas");
const overlay = $("skeleton");
const edges = [[0,1], [0,2], [1,3], [2,4], [5,6], [5,7], [7,9], [6,8],
  [8,10], [5,11], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]];
const halpeEdges = [[17,18],[5,18],[6,18],[18,19],[11,19],[12,19],
  [15,24],[24,20],[20,22],[16,25],[25,21],[21,23]];
const phases = {UP: "Верх", DESCENT: "Опускание", BOTTOM: "Низ", ASCENT: "Подъём"};
const cues = {
  UP: "Начинайте присед из положения стоя",
  DESCENT: "Продолжайте опускаться",
  BOTTOM: "Возвращайтесь в исходное положение",
  ASCENT: "Закончите подъём",
};
let active = null;
let report = null;

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  return [Math.floor(s / 60), s % 60].map(n => String(n).padStart(2, "0")).join(":");
}

function setChoices(open) {
  visible("choices", open);
  $("exercise").setAttribute("aria-expanded", String(open));
}
$("exercise").onclick = () => setChoices($("choices").classList.contains("hidden"));
$("choices").onclick = (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  setChoices(false);
  $("exercise").focus();
  if (button.dataset.exercise !== "squat") {
    $("devTitle").textContent = button.textContent;
    $("development").showModal();
  }
};
document.addEventListener("click", (e) => {
  if (!e.target.closest(".exercise-picker")) setChoices(false);
  if (!e.target.closest(".profile")) document.querySelector(".profile").open = false;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("choices").classList.contains("hidden")) {
    setChoices(false);
    $("exercise").focus();
  }
});
for (const modal of document.querySelectorAll("dialog")) {
  modal.onclick = (e) => {
    if (e.target !== modal) return;
    const box = modal.getBoundingClientRect();
    if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) modal.close();
  };
}

const profile = document.querySelector(".profile-menu");
try {
  const saved = JSON.parse(localStorage.getItem("rightmove.profile") || "{}");
  for (const field of profile.elements) {
    if (field.name && typeof saved[field.name] === "string") field.value = saved[field.name];
  }
} catch { /* A browser may disable local storage. The form still works in this tab. */ }
profile.onsubmit = (e) => {
  e.preventDefault();
  try { localStorage.setItem("rightmove.profile", JSON.stringify(Object.fromEntries(new FormData(profile)))); }
  catch { /* Keep the current form values for this visit. */ }
  document.querySelector(".profile").open = false;
};

function drawPose(frame) {
  const ctx = overlay.getContext("2d");
  if (frame && (overlay.width !== frame.width || overlay.height !== frame.height)) {
    overlay.width = frame.width;
    overlay.height = frame.height;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!frame || !$("showSkeleton").checked) return;
  const points = frame.points;
  ctx.lineWidth = Math.max(2, overlay.width / 210);
  ctx.strokeStyle = "#32aaff";
  ctx.fillStyle = "#74ff9e";
  for (const [a,b] of (points.length === 26 ? edges.concat(halpeEdges) : edges)) {
    if (!points[a] || points[a][2] < 0.5 || points[b][2] < 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  }
  for (const [x,y,confidence] of points) {
    if (confidence < 0.5) continue;
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth * 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updatePreview() {
  $("demo").src = $("showSkeleton").checked ? "img/squat-pose.webp" : "img/squat.webp";
  drawPose(active?.frame);
}
$("showSkeleton").onchange = updatePreview;

async function api(route, options = {}) {
  const response = await fetch("/api/" + route, {
    method: "POST", signal: AbortSignal.timeout(20000), ...options,
    headers: {...options.headers, "X-RightMove-API": API_VERSION},
  });
  if (response.headers.get("X-RightMove-API") !== API_VERSION) {
    throw new Error("Открыта другая версия RightMove. Остановите старый сервер и запустите web_app.py из SquatCounter.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Ошибка локального сервера.");
  return data;
}

function releaseSession(key) {
  if (key) fetch("/api/finish", {method: "POST", headers: {"X-Session": key, "X-RightMove-API": API_VERSION}, keepalive: true}).catch(() => {});
}

function openReport() {
  if (!report) return;
  $("reportReps").textContent = report.reps;
  $("reportShort").textContent = report.partial;
  $("reportTime").textContent = formatTime(report.ms);
  $("reportRating").textContent = report.rating === null ? "—" : `${report.rating}/5`;
  const details = [];
  if (report.mean_score !== null) {
    details.push(`Средний балл — ${report.mean_score.toLocaleString("ru", {maximumFractionDigits: 1})}/100`);
  }
  if (report.not_evaluated) details.push(`Не оценено — ${report.not_evaluated}`);
  if (report.preliminary && report.rating !== null) details.push("Предварительная оценка");
  $("reportDetails").textContent = details.join(" · ");

  $("repRows").replaceChildren();
  for (const attempt of report.attempts) {
    const row = $("repRows").insertRow();
    const comments = attempt.issues.map(({name, status}) => `${name} (${status})`);
    if (attempt.note) comments.push(`Не оценено — ${attempt.note}`);
    if (!comments.length) comments.push("По проверенным правилам замечаний нет");
    if (attempt.unchecked.length) comments.push(`Не оценено — ${attempt.unchecked.join(", ")}`);
    const label = attempt.rep === null
      ? `Попытка ${attempt.attempt} · ${attempt.category}`
      : `№ ${attempt.rep}`;
    const score = attempt.score === null ? "—" : `${attempt.score}/100`;
    for (const value of [label, comments.join(" · "), score]) row.insertCell().textContent = value;
  }
  visible("repTable", report.attempts.length > 0);
  visible("reportEmpty", report.attempts.length === 0);
  if (!$("report").open) $("report").showModal();
}

async function finish(run, message = "", showReport = true) {
  if (active !== run || run.finishing) return;
  run.finishing = true;
  const duration = run.started === null ? 0 : performance.now() - run.started;
  clearInterval(run.clock);
  clearTimeout(run.next);
  run.stream?.getTracks().forEach(track => track.stop());
  camera.srcObject = null;
  for (const id of ["camera", "skeleton", "countdown", "phase", "calibrationPanel", "assessmentPanel", "btnCalibrate"]) visible(id, false);
  visible("demo", true);
  visible("previewLabel", true);
  $("btnStop").disabled = $("btnCalibrate").disabled = true;
  let receivedReport = false;
  if (run.key && run.countdownDone && showReport) {
    $("cue").textContent = "Готовим отчёт…";
    try {
      await run.pending?.catch(() => {});
      await run.recalibration?.catch(() => {});
      const summary = await api("finish", {headers: {"X-Session": run.key}});
      report = {...summary, ms: duration};
      receivedReport = true;
      $("reps").textContent = summary.reps;
      $("timer").textContent = formatTime(duration);
    } catch (err) {
      message = [message, "Не удалось получить отчёт. " + err.message].filter(Boolean).join(" ");
    }
  } else {
    releaseSession(run.key);
  }
  active = null;
  $("btnStop").textContent = "Завершить";
  $("btnStart").disabled = false;
  $("exercise").disabled = false;
  $("btnReport").disabled = !report;
  $("cue").textContent = "Для начала тренировки нажмите «Начать»";
  $("cameraError").textContent = message;
  visible("cameraError", Boolean(message));
  if (receivedReport && !message) openReport();
}

function drawAssessment(frame) {
  visible("calibrationPanel", !frame.ready);
  visible("assessmentPanel", frame.ready);
  if (!frame.ready) {
    $("calibrationProgress").value = frame.calibration;
    $("calibrationPercent").textContent = frame.calibration + "%";
    $("calibrationStatus").textContent = frame.wrong_view
      ? "Ракурс не подходит. Повернитесь боком к камере."
      : "Встаньте боком в полный рост и стойте спокойно.";
    return;
  }
  const last = frame.assessment;
  $("assessmentTitle").textContent = last
    ? [last.rep === null ? last.category : `Повторение № ${last.rep}`, last.outcome].join(" · ")
    : "Калибровка завершена. Ожидание первой попытки.";
  for (const [code, label] of Object.entries(ruleLabels)) {
    const outcome = last?.rules[code] ?? "не оценено";
    const mark = $("rule" + code);
    mark.textContent = label + " — " + outcome;
    mark.dataset.status = ruleStyles[outcome] ?? "unknown";
  }
  $("assessmentScore").textContent = last?.score != null ? `Балл — ${last.score}/100` : last?.note ?? "";
  const summary = frame.summary;
  if (summary) {
    const grade = summary.rating === null ? "—" : `${summary.rating}/5${summary.preliminary ? " (предварительно)" : ""}`;
    $("assessmentSummary").textContent = `Неполных — ${summary.partial} · Не оценено — ${summary.not_evaluated} · Оценка — ${grade}`;
  }
}

function handleFrame(run, frame) {
  run.frame = frame;
  $("reps").textContent = frame.count;
  drawPose(frame);
  drawAssessment(frame);
  if (!frame.ready) {
    visible("phase", false);
    $("cue").textContent = frame.wrong_view && frame.calibration === 0
      ? "Повернитесь боком к камере."
      : `Стойте спокойно боком к камере. Калибровка ${frame.calibration}%`;
    return;
  }
  if (run.started === null) run.started = performance.now();
  const assessment = frame.assessment;
  if (assessment && assessment.attempt !== run.lastAssessment) {
    run.lastAssessment = assessment.attempt;
    run.notice = assessment.issues[0]?.name || assessment.note ||
      (assessment.score === null ? assessment.category : `Оценка повторения — ${assessment.score}/100`);
    run.noticeUntil = performance.now() + 1800;
  }
  visible("phase", frame.angle !== null);
  if (frame.angle !== null) $("phase").textContent = phases[frame.phase] + " · " + Math.round(frame.angle) + "°";
  $("cue").textContent = !frame.sides
    ? "Плечо, бедро, колено и лодыжка должны быть видны сбоку."
    : performance.now() < run.noticeUntil ? run.notice : cues[frame.phase];
}

async function processFrame(run) {
  if (active !== run || run.finishing || run.recalibration) return;
  const epoch = run.frameEpoch;
  try {
    if (camera.readyState >= 2 && camera.currentTime !== run.videoTime) {
      run.videoTime = camera.currentTime;
      const scale = Math.min(1, 640 / Math.max(camera.videoWidth, camera.videoHeight));
      capture.width = Math.max(1, Math.round(camera.videoWidth * scale));
      capture.height = Math.max(1, Math.round(camera.videoHeight * scale));
      capture.getContext("2d").drawImage(camera, 0, 0, capture.width, capture.height);
      const blob = await new Promise(resolve => capture.toBlob(resolve, "image/jpeg", 0.84));
      if (active !== run || run.finishing || epoch !== run.frameEpoch) return;
      if (!blob) throw new Error("Не удалось получить кадр камеры.");
      run.pending = api("frame", {headers: {"X-Session": run.key}, body: blob});
      const frame = await run.pending;
      run.pending = null;
      if (active !== run || run.finishing || epoch !== run.frameEpoch) return;
      handleFrame(run, frame);
    }
    run.next = setTimeout(() => processFrame(run), 16);
  } catch (err) {
    finish(run, "Обработка остановлена. " + err.message);
  }
}

async function start() {
  if (active) return;
  const run = {stream: null, key: null, started: null, countdownDone: false, frameEpoch: 0, noticeUntil: 0};
  active = run;
  setChoices(false);
  $("exercise").disabled = $("btnStart").disabled = $("btnReport").disabled = true;
  $("btnStop").disabled = false;
  $("btnStop").textContent = "Отменить";
  $("cue").textContent = "Разрешите браузеру доступ к камере";
  visible("cameraError", false);
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Откройте сайт по локальному адресу из окна запуска.");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: 960}, height: {ideal: 720}, facingMode: "user"}, audio: false,
    });
    if (active !== run) { stream.getTracks().forEach(track => track.stop()); return; }
    run.stream = stream;
    const track = stream.getVideoTracks()[0];
    track.onended = () => finish(run, "Камера отключена. Подключите её и нажмите «Начать».");
    camera.srcObject = stream;
    await camera.play();
    if (active !== run || run.finishing) return;
    if (track.readyState !== "live") throw new Error("Камера отключена.");
    visible("demo", false);
    visible("previewLabel", false);
    visible("camera", true);
    $("cue").textContent = "Подключение к модели…";
    const {session} = await api("session");
    if (active !== run) { releaseSession(session); return; }
    run.key = session;
    drawPose(null);
    $("reps").textContent = "0";
    $("timer").textContent = "00:00";
    $("countdownNum").textContent = "10";
    visible("countdown", true);
    const deadline = performance.now() + 10000;
    run.clock = setInterval(() => {
      const now = performance.now();
      if (run.started !== null) { $("timer").textContent = formatTime(now - run.started); return; }
      if (run.countdownDone) return;
      const left = Math.ceil((deadline - now) / 1000);
      if (left > 0) { $("countdownNum").textContent = left; return; }
      run.countdownDone = true;
      visible("countdown", false);
      visible("skeleton", true);
      drawAssessment({ready: false, calibration: 0, wrong_view: false});
      visible("btnCalibrate", true);
      $("btnCalibrate").disabled = false;
      $("btnStop").textContent = "Завершить";
      $("cue").textContent = "Встаньте прямо, боком к камере";
      processFrame(run);
    }, 100);
  } catch (err) {
    const reasons = {
      NotAllowedError: "Доступ к камере запрещён. Разрешите его в настройках сайта и снова нажмите «Начать».",
      NotFoundError: "Камера не найдена. Подключите камеру и попробуйте ещё раз.",
      NotReadableError: "Камера недоступна. Проверьте подключение и закройте приложения, которые её используют.",
    };
    finish(run, reasons[err.name] || "Не удалось начать тренировку. " + err.message);
  }
}
$("btnStart").onclick = start;
$("btnStop").onclick = () => { if (active) finish(active); };
$("btnReport").onclick = openReport;
$("btnCalibrate").onclick = async () => {
  const run = active;
  if (!run || run.finishing || run.recalibration) return;
  $("btnCalibrate").disabled = true;
  run.frameEpoch += 1;
  clearTimeout(run.next);
  run.recalibration = (async () => {
    await run.pending;
    if (!run.finishing) await api("recalibrate", {headers: {"X-Session": run.key}});
  })();
  try {
    await run.recalibration;
  } catch (err) {
    run.recalibration = null;
    finish(run, err.message);
    return;
  }
  run.recalibration = null;
  if (active !== run || run.finishing) return;
  drawAssessment({ready: false, calibration: 0, wrong_view: false});
  visible("phase", false);
  $("btnCalibrate").disabled = false;
  processFrame(run);
};
window.addEventListener("pagehide", () => { if (active) finish(active, "", false); });

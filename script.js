const charts = {};

async function api(url) {
  const separator = url.includes("?") ? "&" : "?";
  const response = await fetch(`${url}${separator}_ts=${Date.now()}`, { cache: "no-store" });
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The server returned an invalid response. Check the terminal window."); }
  if (!response.ok) throw new Error(data.error || data.message || `Request failed (HTTP ${response.status})`);
  return data;
}

function setText(id, value) { const element = document.getElementById(id); if (element) element.textContent = value; }
function updateClock() { setText("clock", new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })); }
function displayValue(value, suffix = "") { return value === null || value === undefined ? "N/A" : `${value}${suffix}`; }

async function loadWeather() {
  setText("weatherExtra", "Fetching local conditions…");
  try {
    const data = await api("/api/weather");
    setText("weatherTemp", displayValue(data.temperature, "°C"));
    setText("weatherExtra", `Rain: ${displayValue(data.rain, " mm")}  ·  Wind: ${displayValue(data.wind_speed, " km/h")}  ·  Updated ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    setText("weatherTemp", "Unavailable"); setText("weatherExtra", error.message);
  }
}

async function loadTraffic() {
  setText("trafficStatus", "Checking…"); setText("trafficDetails", "Fetching the latest road segment data.");
  try {
    const data = await api("/api/traffic");
    if (!data.available) { setText("trafficStatus", "Unavailable"); setText("trafficDetails", data.message || "Live traffic is unavailable."); return; }
    setText("trafficStatus", data.traffic_status || "Status unavailable");
    setText("trafficDetails", `Current speed: ${displayValue(data.current_speed_kmph, " km/h")}  ·  Delay: ${displayValue(data.estimated_delay_minutes, " min")}  ·  Updated ${new Date().toLocaleTimeString()}`);
  } catch (error) { setText("trafficStatus", "Connection issue"); setText("trafficDetails", error.message); }
}

async function loadBestTime() {
  setText("bestDetails", "Recalculating from the loaded traffic dataset…");
  try {
    const data = await api("/api/best-time");
    setText("bestTime", data.best_time);
    setText("bestDetails", `${data.congestion} congestion  ·  Score ${data.congestion_score}  ·  ${data.time_period}  ·  ${data.basis}`);
  } catch (error) { setText("bestTime", "Unavailable"); setText("bestDetails", error.message); }
}

async function estimateTrip(event) {
  if (event) event.preventDefault();
  const query = new URLSearchParams({ distance_km: document.getElementById("distance").value, speed_kmph: document.getElementById("speed").value });
  try {
    const data = await api(`/api/trip-time?${query}`);
    setText("tripDuration", data.duration_text);
    setText("tripBasis", `${data.distance_km} km at ${data.speed_kmph} km/h`);
  } catch (error) { setText("tripDuration", "Check input"); setText("tripBasis", error.message); }
}

async function predictCongestion(event) {
  event.preventDefault();
  const result = document.getElementById("predictionResult");
  result.classList.add("is-loading");
  setText("predictionLevel", "Working…"); setText("predictionScore", "Evaluating conditions");
  const query = new URLSearchParams({
    hour: document.getElementById("hour").value,
    traffic_volume: document.getElementById("volume").value,
    temperature: document.getElementById("temp").value,
    rainfall: document.getElementById("rain").value,
    model: document.getElementById("model").value
  });
  try {
    const data = await api(`/api/predict?${query}`);
    setText("predictionLevel", data.congestion);
    setText("predictionScore", data.congestion_score === null ? data.method : `Congestion score: ${data.congestion_score} · ${data.time_period}`);
    setText("recommendation", data.recommendation);
    result.classList.remove("level-low", "level-medium", "level-high");
    result.classList.add(`level-${String(data.congestion).toLowerCase()}`);
  } catch (error) { setText("predictionLevel", "Unavailable"); setText("predictionScore", error.message); setText("recommendation", "Check the inputs and try again."); }
  finally { result.classList.remove("is-loading"); }
}

function chartOptions() {
  return { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { backgroundColor: "#172b4d", padding: 10, cornerRadius: 7 } }, scales: { x: { grid: { display: false }, ticks: { color: "#7888a0", font: { size: 10 } } }, y: { beginAtZero: true, grid: { color: "#edf1f7" }, ticks: { color: "#7888a0", font: { size: 10 } } } } };
}
function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const canvas = document.getElementById(id);
  if (canvas) charts[id] = new Chart(canvas, config);
}

function renderModelMetrics(metrics) {
  const container = document.getElementById("modelMetrics");
  if (!metrics || !metrics.available) { container.textContent = metrics?.message || "Model evaluation is unavailable."; return; }
  const entries = Object.entries(metrics.models || {}).filter(([, model]) => model.accuracy !== undefined);
  if (!entries.length) { container.textContent = "No model evaluation results are available."; return; }
  container.innerHTML = entries.map(([name, model], index) => `<div class="model-stat"><span class="model-symbol">${index === 0 ? "🌳" : "📈"}</span><div><small>${name}</small><strong>${(model.accuracy * 100).toFixed(1)}% accuracy</strong></div></div>`).join("");
}

async function loadCharts() {
  try {
    const data = await api("/api/analytics");
    makeChart("distributionChart", { type: "doughnut", data: { labels: data.distribution.labels, datasets: [{ data: data.distribution.values, backgroundColor: ["#24b879", "#f3a326", "#e64b4b"], borderWidth: 3, borderColor: "#fff", hoverOffset: 5 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: "58%", plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 7, padding: 14, color: "#526580", font: { size: 10 } } } } } });
    makeChart("trafficChart", { type: "line", data: { labels: data.hourly.hours, datasets: [{ label: "Average traffic", data: data.hourly.average_traffic, borderColor: "#2385e8", backgroundColor: "rgba(35,133,232,.12)", fill: true, tension: .35, pointRadius: 2, pointHoverRadius: 5, spanGaps: true }] }, options: chartOptions() });
    makeChart("scoreChart", { type: "line", data: { labels: data.hourly.hours, datasets: [{ label: "Average score", data: data.hourly.average_score, borderColor: "#895bd1", backgroundColor: "rgba(137,91,209,.12)", fill: true, tension: .35, pointRadius: 2, pointHoverRadius: 5, spanGaps: true }] }, options: chartOptions() });
    makeChart("scatterChart", { type: "scatter", data: { datasets: [{ label: "Traffic records", data: data.scatter, backgroundColor: "rgba(28,177,119,.75)", pointRadius: 3, pointHoverRadius: 5 }] }, options: { ...chartOptions(), scales: { x: { ...chartOptions().scales.x, title: { display: true, text: "Traffic volume", color: "#64748b" } }, y: { ...chartOptions().scales.y, title: { display: true, text: "Temperature (°C)", color: "#64748b" } } } } });
    renderModelMetrics(data.model_metrics);
  } catch (error) { document.getElementById("modelMetrics").textContent = error.message; }
}

function bindRefreshButton(id, action) {
  const button = document.getElementById(id);
  button.addEventListener("click", async () => {
    const original = button.textContent;
    button.disabled = true; button.textContent = "Refreshing…";
    try { await action(); } finally { button.disabled = false; button.textContent = original; }
  });
}

function initNavigation() {
  document.querySelectorAll(".nav-link").forEach(link => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active"));
    link.classList.add("active");
  }));
}

document.addEventListener("DOMContentLoaded", () => {
  updateClock(); setInterval(updateClock, 30000);
  document.getElementById("tripForm").addEventListener("submit", estimateTrip);
  document.getElementById("predictionForm").addEventListener("submit", predictCongestion);
  bindRefreshButton("refreshWeatherBtn", loadWeather);
  bindRefreshButton("refreshTrafficBtn", loadTraffic);
  bindRefreshButton("refreshBestBtn", loadBestTime);
  initNavigation();
  loadWeather(); loadTraffic(); loadBestTime(); loadCharts();
  estimateTrip();
});


function bindRefreshButton(id, action, loadingText) {
    const btn = document.getElementById(id);

    if (!btn) {
        console.error("Button not found:", id);
        return;
    }

    btn.addEventListener("click", async () => {
        btn.disabled = true;

        try {
            await action();
        } catch (error) {
            console.error("Refresh failed:", error);
            alert(error.message);
        } finally {
            btn.disabled = false;
        }
    });
}


bindRefreshButton("refreshWeatherBtn", loadWeather, "Refreshing...");
bindRefreshButton("refreshTrafficBtn", loadTraffic, "Refreshing...");
bindRefreshButton("refreshBestBtn", loadBest, "Refreshing...");
bindRefreshButton("estimateTripBtn", estimateTrip, "Estimating...");
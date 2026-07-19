const GRADE_INFO = {
  0: {
    label: "No Diabetic Retinopathy",
    icon: "✅",
    description: "No signs of diabetic retinopathy detected. The retina appears healthy. Continue regular annual eye checkups."
  },
  1: {
    label: "Mild NPDR",
    icon: "🟡",
    description: "Mild Non-Proliferative DR. Microaneurysms are present. Monitor closely and control blood sugar levels."
  },
  2: {
    label: "Moderate NPDR",
    icon: "🟠",
    description: "Moderate Non-Proliferative DR. Blockage in blood vessels detected. Ophthalmologist visit recommended within 3 months."
  },
  3: {
    label: "Severe NPDR",
    icon: "🔴",
    description: "Severe Non-Proliferative DR. Many blocked vessels detected. Urgent ophthalmologist referral required."
  },
  4: {
    label: "Proliferative DR",
    icon: "🟣",
    description: "Proliferative DR — most advanced stage. Abnormal new blood vessels are growing. Immediate medical intervention required."
  }
};

const GRADE_COLORS = {
  0: "#00c864",
  1: "#ffc800",
  2: "#ff8c00",
  3: "#ff3c3c",
  4: "#b400ff"
};

const dropZone          = document.getElementById("dropZone");
const fileInput         = document.getElementById("fileInput");
const previewBox        = document.getElementById("previewBox");
const previewImg        = document.getElementById("previewImg");
const removeBtn         = document.getElementById("removeBtn");
const analyzeBtn        = document.getElementById("analyzeBtn");
const btnText           = document.getElementById("btnText");
const btnLoader         = document.getElementById("btnLoader");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent     = document.getElementById("resultContent");
const gradeNumber       = document.getElementById("gradeNumber");
const gradeBadge        = document.getElementById("gradeBadge");
const severityIcon      = document.getElementById("severityIcon");
const severityText      = document.getElementById("severityText");
const confidenceValue   = document.getElementById("confidenceValue");
const confidenceFill    = document.getElementById("confidenceFill");
const resultDescription = document.getElementById("resultDescription");
const resetBtn          = document.getElementById("resetBtn");

let selectedFile = null;

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    handleFile(file);
  }
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewBox.style.display = "block";
    dropZone.style.display = "none";
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

removeBtn.addEventListener("click", () => {
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  previewImg.src = "";
  previewBox.style.display = "none";
  dropZone.style.display = "block";
  analyzeBtn.disabled = true;
  fileInput.value = "";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  btnText.style.display = "none";
  btnLoader.style.display = "inline";
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    showResult(data.grade, data.confidence, data.all_probabilities);
    addToHistory(data.grade, data.confidence);

  } catch (error) {
    alert("Error: Could not connect to the server.\nMake sure app.py is running.");
  } finally {
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
    analyzeBtn.disabled = false;
  }
});

function showResult(grade, confidence, allProbs) {
  const info = GRADE_INFO[grade];
  const color = GRADE_COLORS[grade];

  gradeNumber.textContent = grade;
  gradeBadge.style.borderColor = color;
  gradeBadge.style.background = `${color}18`;
  gradeNumber.style.color = color;

  severityIcon.textContent = info.icon;
  severityText.textContent = info.label;
  severityText.style.color = color;

  const pct = Math.round(confidence * 100);
  confidenceValue.textContent = `${pct}%`;
  confidenceFill.style.width = `${pct}%`;
  confidenceFill.style.background = `linear-gradient(90deg, ${color}, ${color}99)`;

  resultDescription.textContent = info.description;

  renderChart(allProbs);

  resultPlaceholder.style.display = "none";
  resultContent.style.display = "block";
}

function renderChart(probs) {
  const labels = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"];
  const chartBars = document.getElementById("chartBars");
  chartBars.innerHTML = "";

  probs.forEach((prob, i) => {
    const color = GRADE_COLORS[i];
    const pct = Math.round(prob * 100);
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
      <span class="chart-label">${labels[i]}</span>
      <div class="chart-bar-wrap">
        <div class="chart-bar-fill" style="width:${pct}%; background:${color};"></div>
      </div>
      <span class="chart-pct">${pct}%</span>
    `;
    chartBars.appendChild(row);
  });
}

function addToHistory(grade, confidence) {
  const historySection = document.getElementById("historySection");
  const historyGrid   = document.getElementById("historyGrid");
  const color = GRADE_COLORS[grade];
  const info  = GRADE_INFO[grade];
  const now   = new Date().toLocaleTimeString();

  const card = document.createElement("div");
  card.className = "history-card";
  card.innerHTML = `
    <img src="${previewImg.src}" alt="Scan" />
    <div class="history-info">
      <div class="history-grade" style="color:${color}">Grade ${grade} — ${Math.round(confidence * 100)}%</div>
      <div class="history-label">${info.label}</div>
      <div class="history-time">${now}</div>
    </div>
  `;

  historyGrid.prepend(card);
  historySection.style.display = "block";
}

resetBtn.addEventListener("click", () => {
  resetUpload();
  resultContent.style.display = "none";
  resultPlaceholder.style.display = "flex";
});

(function () {
  "use strict";

  var ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  var MAX_BYTES = 12 * 1024 * 1024; // 12MB, generous for a free-tier upload

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var dzIdle = document.getElementById("dz-idle");
  var dzPreview = document.getElementById("dz-preview");
  var previewImg = document.getElementById("dz-preview-img");
  var previewName = document.getElementById("dz-filename");
  var previewSize = document.getElementById("dz-filesize");
  var clearBtn = document.getElementById("dz-clear");

  var form = document.getElementById("upload-form");
  var runBtn = document.getElementById("run-btn");
  var errorBox = document.getElementById("form-error");

  var resultsPanel = document.getElementById("results-panel");
  var imgOriginal = document.getElementById("img-original");
  var imgAnnotated = document.getElementById("img-annotated");
  var figOriginal = document.getElementById("fig-original");
  var figAnnotated = document.getElementById("fig-annotated");
  var imageGrid = document.getElementById("image-grid");
  var countPill = document.getElementById("count-pill");
  var latencyPill = document.getElementById("latency-pill");
  var detectionList = document.getElementById("detection-list");
  var resetBtn = document.getElementById("reset-btn");
  var viewToggleBtns = document.querySelectorAll(".view-toggle button");
  var sampleTiles = document.querySelectorAll(".sample-tile");

  var selectedFile = null;

  function humanSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = msg ? "block" : "none";
  }

  function setSelectedFile(file) {
    showError("");

    if (!file) {
      selectedFile = null;
      dzIdle.style.display = "";
      dzPreview.style.display = "none";
      dropzone.classList.remove("has-file");
      runBtn.disabled = true;
      return;
    }

    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
      showError("Unsupported file type. Please use JPG, PNG, or WEBP.");
      setSelectedFile(null);
      return;
    }

    if (file.size > MAX_BYTES) {
      showError("File is too large (max 12MB).");
      setSelectedFile(null);
      return;
    }

    selectedFile = file;
    dzIdle.style.display = "none";
    dzPreview.style.display = "flex";
    dropzone.classList.add("has-file");
    previewName.textContent = file.name;
    previewSize.textContent = humanSize(file.size);

    var reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    runBtn.disabled = false;
  }

  // click-to-browse (the <input> also sits on top of the dropzone for native support)
  fileInput.addEventListener("change", function () {
    setSelectedFile(fileInput.files[0] || null);
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", function (e) {
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files[0]) {
      fileInput.files = files;
      setSelectedFile(files[0]);
    }
  });

  clearBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    fileInput.value = "";
    setSelectedFile(null);
  });

  function setLoading(isLoading) {
    runBtn.disabled = isLoading;
    runBtn.classList.toggle("loading", isLoading);
  }

  function renderDetections(predictions) {
    detectionList.innerHTML = "";

    if (!predictions.length) {
      var empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No heads detected in this image.";
      detectionList.appendChild(empty);
      return;
    }

    predictions.forEach(function (p) {
      var badge = document.createElement("span");
      badge.className = "det-badge";
      badge.innerHTML =
        '<span class="swatch"></span>Head detected' +
        '<span class="conf">' + (p.confidence * 100).toFixed(1) + "%</span>";
      detectionList.appendChild(badge);
    });
  }

  function setView(view) {
    viewToggleBtns.forEach(function (btn) {
      btn.classList.toggle("active", btn.dataset.view === view);
    });
    if (view === "both") {
      imageGrid.classList.remove("single-view");
    } else {
      imageGrid.classList.add("single-view");
      figOriginal.classList.toggle("active-view", view === "original");
      figAnnotated.classList.toggle("active-view", view === "annotated");
    }
  }

  viewToggleBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      setView(btn.dataset.view);
    });
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!selectedFile) return;

    showError("");
    setLoading(true);

    var formData = new FormData();
    formData.append("image", selectedFile);

    var startedAt = performance.now();

    fetch("/api/predict", { method: "POST", body: formData })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || "Detection failed.");
          return data;
        });
      })
      .then(function (data) {
        var elapsedMs = Math.round(performance.now() - startedAt);

        imgOriginal.src = data.uploaded_image;
        imgAnnotated.src = data.annotated_image;
        countPill.textContent = data.prediction_count + " head" + (data.prediction_count === 1 ? "" : "s") + " detected";
        latencyPill.textContent = "~" + elapsedMs + "ms round-trip";
        renderDetections(data.predictions || []);
        setView("both");

        showResultsPanel();
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      })
      .catch(function (err) {
        showError(err.message || "Something went wrong. Please try again.");
      })
      .finally(function () {
        setLoading(false);
      });
  });

  // Toggling display:none straight to block skips the CSS transition, so we
  // flip it in two frames: let the browser commit the layout change, then
  // add the class that actually animates in.
  function showResultsPanel() {
    resultsPanel.style.display = "block";
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        resultsPanel.classList.add("visible");
      });
    });
  }

  function hideResultsPanel() {
    resultsPanel.classList.remove("visible");
    window.setTimeout(function () {
      resultsPanel.style.display = "none";
    }, 500);
  }

  resetBtn.addEventListener("click", function () {
    fileInput.value = "";
    setSelectedFile(null);
    hideResultsPanel();
    showError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Sample images: fetch the static file, turn it into a File, and feed it
  // through the same path a real drag-drop/browse selection would take.
  function loadSampleImage(tile) {
    var src = tile.dataset.src;
    var name = tile.dataset.name || "sample.jpg";

    showError("");
    fetch(src)
      .then(function (res) { return res.blob(); })
      .then(function (blob) {
        var file = new File([blob], name, { type: blob.type });
        var dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        setSelectedFile(file);
        dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
      })
      .catch(function () {
        showError("Couldn't load that sample image. Please try uploading your own.");
      });
  }

  sampleTiles.forEach(function (tile) {
    tile.addEventListener("click", function () {
      loadSampleImage(tile);
    });
  });

  // Scroll-triggered reveal: sections marked ".reveal" pop into place
  // the first time they cross into the viewport.
  var revealTargets = document.querySelectorAll(".reveal");
  if (revealTargets.length) {
    if ("IntersectionObserver" in window) {
      var revealObserver = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add("in-view");
              revealObserver.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.2, rootMargin: "0px 0px -40px 0px" }
      );
      revealTargets.forEach(function (el) { revealObserver.observe(el); });
    } else {
      revealTargets.forEach(function (el) { el.classList.add("in-view"); });
    }
  }

  // initial state
  setSelectedFile(null);
})();

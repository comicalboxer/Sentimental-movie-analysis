document.addEventListener("DOMContentLoaded", () => {
  const reviewInput = document.getElementById("reviewInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const charCount = document.querySelector(".char-count");
  const resultSection = document.getElementById("resultSection");
  const btnText = document.querySelector(".btn-text");
  const btnLoader = document.querySelector(".btn-loader");

  // Result elements
  const sentimentLabel = document.getElementById("sentimentLabel");
  const sentimentDesc = document.getElementById("sentimentDesc");
  const scoreValue = document.getElementById("scoreValue");
  const gaugeFill = document.getElementById("gaugeFill");

  // Char count update
  reviewInput.addEventListener("input", () => {
    charCount.textContent = `${reviewInput.value.length} characters`;
  });

  analyzeBtn.addEventListener("click", async () => {
    const review = reviewInput.value.trim();

    if (!review) {
      alert("Please enter a review first!");
      return;
    }

    // Loading state
    setLoading(true);
    resultSection.classList.add("hidden");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review: review }),
      });

      const data = await response.json();

      if (response.ok) {
        showResult(data);
      } else {
        alert(`Error: ${data.error || "Something went wrong"}`);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to connect to the server.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
      btnText.classList.add("hidden");
      btnLoader.classList.remove("hidden");
    } else {
      btnText.classList.remove("hidden");
      btnLoader.classList.add("hidden");
    }
  }

  function showResult(data) {
    resultSection.classList.remove("hidden");

    const sentiment = data.sentiment;
    const score = data.score; // 0 to 1
    const percentage = Math.round(score * 100);

    sentimentLabel.textContent = sentiment;
    scoreValue.textContent = `${percentage}%`;

    // Gauge animation logic
    // 10 to 90 is the path length for 180 degrees approx in this specific SVG Setup ~ 125.6px
    const maxDash = 126;
    const offset = maxDash - (maxDash * score);
    gaugeFill.style.strokeDashoffset = offset;

    if (sentiment === "Positive") {
      sentimentLabel.className = "positive-glow";
      sentimentDesc.textContent = "Great! This looks like a positive review.";
      gaugeFill.setAttribute("stroke", "#22c55e");
    } else {
      sentimentLabel.className = "negative-glow";
      sentimentDesc.textContent = "Oof. This looks like a negative review.";
      gaugeFill.setAttribute("stroke", "#ef4444");
    }
  }
});

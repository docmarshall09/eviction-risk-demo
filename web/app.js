(function initDemoSite() {
  setActiveNavLink();
  bindScoreForm();
})();

function setActiveNavLink() {
  const activePage = document.body.dataset.page;
  if (!activePage) {
    return;
  }

  const navLink = document.querySelector(`[data-nav="${activePage}"]`);
  if (!navLink) {
    return;
  }

  navLink.classList.add("active");
  navLink.setAttribute("aria-current", "page");
}

function bindScoreForm() {
  const form = document.getElementById("score-form");
  if (!form) {
    return;
  }

  const countyInput = document.getElementById("county_fips");
  const yearInput = document.getElementById("as_of_year");
  const scoreButton = document.getElementById("score-button");
  const resultsCard = document.getElementById("results-card");
  const errorCard = document.getElementById("error-card");
  bindExampleButtons(countyInput, yearInput);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideError(errorCard);

    const countyFips = countyInput.value.trim();
    const asOfYearText = yearInput.value.trim();

    const validationError = validateInputs(countyFips, asOfYearText);
    if (validationError) {
      showError(errorCard, 400, validationError);
      resultsCard.classList.add("hidden");
      return;
    }

    const payload = { county_fips: countyFips };
    if (asOfYearText !== "") {
      payload.as_of_year = Number(asOfYearText);
    }

    setLoadingState(scoreButton, true);

    try {
      const response = await fetch("/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const body = await tryParseJson(response);
      if (!response.ok) {
        const apiError = parseApiError(response.status, body);
        showError(errorCard, apiError.status, apiError.message);
        resultsCard.classList.add("hidden");
        return;
      }

      renderScoreResult(body);
      resultsCard.classList.remove("hidden");
    } catch (error) {
      showError(errorCard, "network", getNetworkMessage(error));
      resultsCard.classList.add("hidden");
    } finally {
      setLoadingState(scoreButton, false);
    }
  });
}

function bindExampleButtons(countyInput, yearInput) {
  const exampleButtons = document.querySelectorAll(".example-button");
  if (!exampleButtons.length) {
    return;
  }

  exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      countyInput.value = button.dataset.exampleFips || "";
      yearInput.value = button.dataset.exampleYear || "";
      countyInput.focus();
    });
  });
}

function validateInputs(countyFips, asOfYearText) {
  if (!/^\d{5}$/.test(countyFips)) {
    return "county_fips must be exactly 5 digits.";
  }

  if (asOfYearText !== "" && !/^\d{4}$/.test(asOfYearText)) {
    return "as_of_year must be a 4-digit year when provided.";
  }

  return null;
}

function setLoadingState(button, isLoading) {
  button.disabled = isLoading;
  button.textContent = isLoading ? "Scoring..." : "Score";
}

function parseApiError(statusCode, body) {
  let message = "Request failed.";
  const detail = body && body.detail ? body.detail : null;

  if (typeof detail === "string") {
    message = detail;
  } else if (Array.isArray(detail)) {
    message = detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  } else if (detail && typeof detail === "object") {
    message = detail.message || JSON.stringify(detail);

    if (
      Number.isInteger(detail.available_years_min) &&
      Number.isInteger(detail.available_years_max)
    ) {
      message += ` Available years: ${detail.available_years_min}-${detail.available_years_max}.`;
    }
  } else if (body && typeof body.message === "string") {
    message = body.message;
  }

  return { status: statusCode, message };
}

async function tryParseJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function getNetworkMessage(error) {
  if (error && typeof error.message === "string") {
    return error.message;
  }
  return "Unable to reach the API. Check that the local server is running.";
}

function showError(errorCard, statusCode, message) {
  const statusElement = document.getElementById("error-status");
  const messageElement = document.getElementById("error-message");

  statusElement.textContent = String(statusCode);
  messageElement.textContent = message;
  errorCard.classList.remove("hidden");
}

function hideError(errorCard) {
  errorCard.classList.add("hidden");
}

function renderScoreResult(payload) {
  const percentilePrimaryElement = document.getElementById("metric-risk-percentile-primary");
  const percentileExplainerElement = document.getElementById("metric-percentile-explainer");
  const riskScoreElement = document.getElementById("metric-risk-score");
  const yearUsedElement = document.getElementById("metric-year-used");
  const yearNoteElement = document.getElementById("metric-year-note");
  const metaElement = document.getElementById("result-meta");
  const rawFieldsElement = document.getElementById("raw-fields-json");
  const featuresElement = document.getElementById("features-json");
  const notesElement = document.getElementById("notes-callout");

  const percentileValue = Number(payload.risk_percentile_in_year);
  const asOfYear = formatInteger(payload.as_of_year);
  const percentileText = formatPercentile(percentileValue);

  percentilePrimaryElement.textContent = percentileText;
  percentileExplainerElement.textContent = buildPercentileExplainer(percentileValue, asOfYear);
  riskScoreElement.textContent = formatNumber(payload.risk_score, 3);
  yearUsedElement.textContent = asOfYear;
  yearNoteElement.textContent = "Leave year blank to use the latest available year for that county.";

  metaElement.textContent = [`County ${payload.county_fips}`, `Compared within ${asOfYear}`].join(" • ");

  // Keep raw API fields in Details so the default view stays non-technical.
  const rawFields = {
    county_fips: payload.county_fips,
    as_of_year: payload.as_of_year,
    as_of_year_available: payload.as_of_year_available,
    available_years_min: payload.available_years_min,
    available_years_max: payload.available_years_max,
    risk_percentile_in_year: payload.risk_percentile_in_year,
    risk_score: payload.risk_score,
    model_version: payload.model_version,
    model_type: payload.model_type,
    notes: payload.notes,
  };

  rawFieldsElement.textContent = JSON.stringify(rawFields, null, 2);
  featuresElement.textContent = JSON.stringify(payload.features_used || {}, null, 2);

  if (payload.notes && String(payload.notes).trim() !== "") {
    notesElement.textContent = payload.notes;
    notesElement.classList.remove("hidden");
  } else {
    notesElement.textContent = "";
    notesElement.classList.add("hidden");
  }
}

function buildPercentileExplainer(percentileValue, asOfYearText) {
  if (Number.isNaN(percentileValue)) {
    return "Percentile could not be computed for this response.";
  }
  return `This is higher than about ${percentileValue.toFixed(1)}% of counties scored for ${asOfYearText}.`;
}

function formatPercentile(percentileValue) {
  if (Number.isNaN(percentileValue)) {
    return "-";
  }
  return `${percentileValue.toFixed(1)} percentile`;
}

function formatNumber(value, digits) {
  const numberValue = Number(value);
  if (Number.isNaN(numberValue)) {
    return "-";
  }
  return numberValue.toFixed(digits);
}

function formatInteger(value) {
  const numberValue = Number(value);
  if (Number.isNaN(numberValue)) {
    return "-";
  }
  return String(Math.trunc(numberValue));
}

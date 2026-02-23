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
  const riskScoreElement = document.getElementById("metric-risk-score");
  const percentileElement = document.getElementById("metric-risk-percentile");
  const asOfYearAvailableElement = document.getElementById("metric-year-available");
  const minYearElement = document.getElementById("metric-years-min");
  const maxYearElement = document.getElementById("metric-years-max");
  const metaElement = document.getElementById("result-meta");
  const featuresElement = document.getElementById("features-json");
  const notesElement = document.getElementById("notes-callout");

  riskScoreElement.textContent = formatNumber(payload.risk_score, 3);
  percentileElement.textContent = formatNumber(payload.risk_percentile_in_year, 1);
  asOfYearAvailableElement.textContent = formatBoolean(payload.as_of_year_available);
  minYearElement.textContent = formatInteger(payload.available_years_min);
  maxYearElement.textContent = formatInteger(payload.available_years_max);

  metaElement.textContent = [
    `county_fips ${payload.county_fips}`,
    `as_of_year ${formatInteger(payload.as_of_year)}`,
    payload.model_type || "unknown_model_type",
    payload.model_version || "unknown_model_version",
  ].join(" • ");

  featuresElement.textContent = JSON.stringify(payload.features_used || {}, null, 2);

  if (payload.notes && String(payload.notes).trim() !== "") {
    notesElement.textContent = payload.notes;
    notesElement.classList.remove("hidden");
  } else {
    notesElement.textContent = "";
    notesElement.classList.add("hidden");
  }
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

function formatBoolean(value) {
  if (value === true) {
    return "true";
  }
  if (value === false) {
    return "false";
  }
  return "-";
}

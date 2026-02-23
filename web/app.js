const COUNTY_LOOKUP_PATH = "/demo/assets/county_fips_lookup.json";
const FINDER_MIN_QUERY_LENGTH = 2;
const FINDER_MAX_RESULTS = 12;

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
  bindCountyFinder(countyInput, yearInput);
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

function bindCountyFinder(countyInput, yearInput) {
  const finderSection = document.getElementById("county-finder-section");
  const finderInput = document.getElementById("county-finder-input");
  const resultsContainer = document.getElementById("county-finder-results");
  if (!finderSection || !finderInput || !resultsContainer) {
    return;
  }

  let countyLookup = [];
  let lookupLoaded = false;
  let lookupFailed = false;

  const closeResults = () => {
    resultsContainer.classList.add("hidden");
    resultsContainer.textContent = "";
    finderInput.setAttribute("aria-expanded", "false");
  };

  const openResults = () => {
    resultsContainer.classList.remove("hidden");
    finderInput.setAttribute("aria-expanded", "true");
  };

  const renderEmptyState = (message) => {
    resultsContainer.textContent = "";
    const emptyItem = document.createElement("p");
    emptyItem.className = "finder-empty";
    emptyItem.textContent = message;
    resultsContainer.appendChild(emptyItem);
    openResults();
  };

  const renderMatchResults = (matches) => {
    resultsContainer.textContent = "";

    matches.forEach((record) => {
      const rowButton = document.createElement("button");
      rowButton.type = "button";
      rowButton.className = "finder-result";
      rowButton.dataset.fips = record.fips;
      rowButton.dataset.selectedName = `${record.county_name}, ${record.state_abbr}`;
      rowButton.textContent = `${formatCountyResultName(record.county_name)}, ${record.state_abbr} — ${record.fips}`;
      resultsContainer.appendChild(rowButton);
    });

    openResults();
  };

  const ensureLookupLoaded = async () => {
    if (lookupLoaded || lookupFailed) {
      return;
    }

    try {
      const response = await fetch(COUNTY_LOOKUP_PATH);
      if (!response.ok) {
        throw new Error(`County lookup fetch failed with status ${response.status}`);
      }

      const payload = await response.json();
      if (!Array.isArray(payload)) {
        throw new Error("County lookup payload is not an array.");
      }

      countyLookup = payload
        .filter(
          (record) =>
            record &&
            typeof record.county_name === "string" &&
            typeof record.state_abbr === "string" &&
            typeof record.fips === "string",
        )
        .map((record) => ({
          county_name: record.county_name.trim(),
          state_abbr: record.state_abbr.trim(),
          fips: record.fips.trim(),
          normalized_county_name: normalizeFinderText(record.county_name),
        }));

      lookupLoaded = true;
    } catch (error) {
      lookupFailed = true;
      console.error(error);
    }
  };

  const updateResultsForQuery = async () => {
    const rawQuery = finderInput.value.trim();
    if (rawQuery.length < FINDER_MIN_QUERY_LENGTH) {
      closeResults();
      return;
    }

    await ensureLookupLoaded();
    if (lookupFailed) {
      renderEmptyState("County lookup is unavailable right now.");
      return;
    }

    const normalizedQuery = normalizeFinderText(rawQuery);
    if (normalizedQuery.length < FINDER_MIN_QUERY_LENGTH) {
      closeResults();
      return;
    }

    const startsWithMatches = [];
    const containsMatches = [];
    for (const record of countyLookup) {
      if (record.normalized_county_name.startsWith(normalizedQuery)) {
        startsWithMatches.push(record);
      } else if (record.normalized_county_name.includes(normalizedQuery)) {
        containsMatches.push(record);
      }
    }

    const orderedMatches = [...startsWithMatches, ...containsMatches].slice(0, FINDER_MAX_RESULTS);
    if (orderedMatches.length === 0) {
      renderEmptyState("No matches found.");
      return;
    }

    renderMatchResults(orderedMatches);
  };

  finderInput.addEventListener("input", () => {
    updateResultsForQuery();
  });

  finderInput.addEventListener("focus", () => {
    if (finderInput.value.trim().length >= FINDER_MIN_QUERY_LENGTH) {
      updateResultsForQuery();
    }
  });

  finderInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeResults();
      finderInput.blur();
    }
  });

  resultsContainer.addEventListener("click", (event) => {
    const selectedButton = event.target.closest(".finder-result");
    if (!(selectedButton instanceof HTMLElement)) {
      return;
    }

    countyInput.value = selectedButton.dataset.fips || "";
    finderInput.value = selectedButton.dataset.selectedName || finderInput.value;
    closeResults();
    yearInput.focus();
  });

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Node)) {
      return;
    }
    if (finderSection.contains(event.target)) {
      return;
    }
    closeResults();
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

function formatCountyResultName(countyName) {
  const trimmedName = countyName.trim();
  const hasCountySuffix = /(county|parish|borough|census area|municipality|city and borough|city|municipio)\b/i.test(
    trimmedName,
  );
  if (hasCountySuffix) {
    return trimmedName;
  }
  return `${trimmedName} County`;
}

function normalizeFinderText(value) {
  return String(value || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
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

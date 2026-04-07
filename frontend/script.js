const predictBtn = document.getElementById("predictBtn");
const imageInput = document.getElementById("imageInput");
const resultBox = document.getElementById("result");

const API_BASE_URL = (localStorage.getItem("API_BASE_URL") || "http://127.0.0.1:5000").replace(/\/$/, "");
const API_URL = `${API_BASE_URL}/predict`;

predictBtn.addEventListener("click", async () => {
	const file = imageInput.files[0];

	if (!file) {
		resultBox.textContent = "Please select an image first.";
		return;
	}

	const formData = new FormData();
	formData.append("image", file);

	predictBtn.disabled = true;
	resultBox.textContent = "Predicting...";

	try {
		const response = await fetch(API_URL, {
			method: "POST",
			body: formData,
		});

		const data = await response.json();

		if (!response.ok) {
			resultBox.textContent = `Error: ${data.error || "Prediction failed"}`;
			return;
		}

		const score = typeof data.score === "number" ? data.score : data.confidence;
		const scorePercent = typeof data.score_percent === "number"
			? data.score_percent.toFixed(2)
			: ((score || 0) * 100).toFixed(2);

		resultBox.textContent = `Predicted Breed: ${data.predicted_class} | Score: ${scorePercent}%`;
	} catch (error) {
		resultBox.textContent = `Error: ${error.message}`;
	} finally {
		predictBtn.disabled = false;
	}
});

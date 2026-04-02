const predictBtn = document.getElementById("predictBtn");
const imageInput = document.getElementById("imageInput");
const resultBox = document.getElementById("result");

const API_URL = "http://127.0.0.1:5000/predict";

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

		const confidencePercent = (data.confidence * 100).toFixed(2);
		resultBox.textContent = `Predicted Breed: ${data.predicted_class} (Confidence: ${confidencePercent}%)`;
	} catch (error) {
		resultBox.textContent = `Error: ${error.message}`;
	} finally {
		predictBtn.disabled = false;
	}
});

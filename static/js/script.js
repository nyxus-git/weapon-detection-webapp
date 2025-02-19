document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let fileInput = document.getElementById("imageUpload").files[0];
    if (!fileInput) {
        alert("Please upload an image!");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);

    let response = await fetch("/predict/", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    document.getElementById("result").innerHTML = `
        <h3>Detection Results</h3>
        <p><b>Detected Objects:</b> ${JSON.stringify(result.detections)}</p>
        <img src="${result.file_path}" width="300px">
    `;
});

function startWebcam() {
    let video = document.getElementById("webcam");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error("Error accessing webcam:", error);
        });
}

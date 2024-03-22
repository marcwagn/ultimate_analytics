const startDrawing = () => {

    const videouploader = document.querySelector("#videoUploader");
    const video = document.querySelector("#videoPlayer");
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");

    videouploader.addEventListener('change', function(event) {
        var file = event.target.files[0];
        var url = URL.createObjectURL(file);

        // send file to flask backend
        var formData = new FormData();
        formData.append('file', file);

        fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => console.log(data.message))
      .catch(error => console.error(error));

        //display video
        video.src = url;
        video.style.display = 'block';

        //let width = canvas.width;
        //let height = canvas.height;
        //const updateCanvas = () => {
        //  ctx.drawImage(video, 0, 0, width, height);
        //  video.requestVideoFrameCallback(updateCanvas);
        //};
        //video.requestVideoFrameCallback(updateCanvas);

    });
}

window.addEventListener('load', startDrawing);

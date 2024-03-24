const startDrawing = () => {

    const videouploader = document.querySelector("#video-uploader");
    const video = document.querySelector("#videoplayer");
    const tacticalBoard = document.querySelector("#tacticalboard");
    //const drawable = document.getElementById('drawable');
    const tacticalBoard_ctx = tacticalBoard.getContext("2d");

    videouploader.addEventListener('change', function(event) {
        var file = event.target.files[0];
        var url = URL.createObjectURL(file);

        // send file to flask backend
        var formData = new FormData();
        formData.append('file', file);

        fetch('https://localhost:5000/upload', {
            mode: "cors",
            method: 'POST',
            body: formData
      })
      .then(response => response.json())
      .then(data => console.log(data.message))
      .catch(error => console.error(error));

        //display video
        video.src = url;
        video.style.display = 'block';

        //show hidden elements
        document.getElementById('dashboard-container').classList.remove('hidden');


        let width = tacticalBoard.width;
        let height = tacticalBoard.height;

        const updateCanvas = () => {
            tacticalBoard_ctx.drawImage(video, 0, 0, width, height);
            video.requestVideoFrameCallback(updateCanvas);
        };
        video.requestVideoFrameCallback(updateCanvas);

    });
}

window.addEventListener('load', startDrawing);

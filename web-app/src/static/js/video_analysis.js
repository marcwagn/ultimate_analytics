const taskForm = (formName, doPoll, report) => {
    document.forms[formName].addEventListener("submit", (event) => {
      event.preventDefault()

      const fileInput = document.querySelector('input[type="file"]');
      const video = document.querySelector("#videoplayer");
      
      const formData = new FormData(event.target)
      formData.append('file', fileInput.files[0]);

      var url = URL.createObjectURL(fileInput.files[0]);
      video.src = url;
      video.style.display = 'block';

      fetch(event.target.action, {
        method: "POST",
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          report(null)

          const poll = () => {
            fetch(`/tasks/result/${data["result_id"]}`)
              .then(response => response.json())
              .then(data => {
                report(data)

                if (!data["ready"]) {
                  setTimeout(poll, 500)
                } else if (!data["successful"]) {
                  console.error(formName, data)
                }
              })
          }

          if (doPoll) {
            poll()
          }
        })
    })
}

taskForm("video-upload-form", true, data => {
  const progressbar = document.getElementById("progressbar")
  if (data === null) {
    console.log("uploading...")
  } else if (!data["ready"]) {
    progressbar.value = data["value"]["status"]
  } else if (!data["successful"]) {
    console.log("error, check console")
  } else {
    progressbar.value = 1;
    document.getElementById('dashboard-container').classList.remove('hidden')
  }
  console.log(data)
})
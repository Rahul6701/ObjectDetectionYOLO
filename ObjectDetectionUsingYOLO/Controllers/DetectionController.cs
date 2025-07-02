using Microsoft.AspNetCore.Mvc;
using ObjectDetectionUsingYOLO.Services;
using SixLabors.ImageSharp;

namespace ObjectDetectUsingONNX.Controllers
{
    public class DetectionController : Controller
    {
        private readonly IWebHostEnvironment _env;
        private readonly OnnxDetectionService _onnxService;

        public DetectionController(IWebHostEnvironment env)
        {
            _env = env;
            var modelPath = Path.Combine(_env.WebRootPath, "AI-Model", "yolov5.onnx");
            //var modelPath = Path.Combine(builder.Environment.WebRootPath, "models", "best.onnx");
            _onnxService = new OnnxDetectionService(modelPath);
        }
        public IActionResult Index()
        {
            return View();
        }
        [HttpPost]
        public IActionResult Upload(IFormFile file)
        {
            // Checks if file is null or empty
            if (file == null || file.Length == 0)
                return BadRequest("No file.");

            var fileName = Guid.NewGuid() + Path.GetExtension(file.FileName);      // Generates a unique filename using a GUID.
            var filePath = Path.Combine(_env.WebRootPath, "uploads", fileName);    // Saves file in wwwroot/uploads.

            // Saves uploaded file to disk.
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                file.CopyTo(stream);
            }

            var detections = _onnxService.Predict(filePath, out var imageWithBoxes);  // Calls your ONNX service to run prediction.

            var resultFileName = "result_" + fileName;
            var resultPath = Path.Combine(_env.WebRootPath, "uploads", resultFileName);
            imageWithBoxes.Save(resultPath);

            ViewBag.ImagePath = "/uploads/" + resultFileName;
            ViewBag.Detections = detections;

            return View("Result");
        }
    }

}

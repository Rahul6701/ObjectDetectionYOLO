using Microsoft.AspNetCore.Mvc;

namespace ObjectDetectionUsingYOLO.Controllers
{
    public class DetectionController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        public IActionResult upload()
        {
            return View();
        }
    }
}

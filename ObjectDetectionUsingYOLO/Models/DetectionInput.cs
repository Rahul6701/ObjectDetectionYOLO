namespace ObjectDetectionUsingYOLO.Models
{
    public class DetectionInput
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
        public float[] Box { get; set; } // [x1, y1, x2, y2]

    }
}

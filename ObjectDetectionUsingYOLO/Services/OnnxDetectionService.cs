using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.Fonts;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using ObjectDetectionUsingYOLO.Models;

namespace ObjectDetectionUsingYOLO.Services
{
    public class OnnxDetectionService
    {
        private readonly InferenceSession _session;
        private readonly int _numClasses = 80;
        private readonly string[] _labels = new[]
        {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        };

        public OnnxDetectionService(string modelPath)
        {
            _session = new InferenceSession(modelPath);
        }

        public List<DetectionInput> Predict(string imagePath, out Image<Rgba32> imageWithBoxes)
        {
            using var image = Image.Load<Rgb24>(imagePath); // Loads image from file path.
            var originalWidth = image.Width;
            var originalHeight = image.Height;

            // Defines YOLO input size (640×640)
            const int inputWidth = 640;
            const int inputHeight = 640;

            // Resizes image to model input size.
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(inputWidth, inputHeight),
                Mode = ResizeMode.Stretch  //  distorts if aspect ratio does not match
            }));

            var tensor = new DenseTensor<float>(new[] { 1, 3, inputHeight, inputWidth });  // Prepares empty tensor for ONNX input shape

            // Loops over each pixel
            // Normalizes RGB values to [0, 1]
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < inputHeight; y++)  // InputHeight = 640
                {
                    var pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < inputWidth; x++)  // // InputWidth = 640 
                    {
                        tensor[0, 0, y, x] = pixelSpan[x].R / 255.0f;
                        tensor[0, 1, y, x] = pixelSpan[x].G / 255.0f;
                        tensor[0, 2, y, x] = pixelSpan[x].B / 255.0f;
                    }
                }
            });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", tensor)
            };

            using var results = _session.Run(inputs);

            var output = results.First().AsEnumerable<float>().ToArray();
            var rawDetections = ParseDetections(output, 0.25f);

            // Applies NMS (Non-Maximum Suppression) to remove duplicate overlapping boxes.
            var finalDetections = ApplyNms(rawDetections, 0.45f);

            // Reloads original image for drawing (preserves original size)
            var drawImage = Image.Load<Rgba32>(imagePath);

            // Creates font for labels.
            var font = SystemFonts.CreateFont("Arial", 20);

            // Converts predicted boxes from model scale to original image scale.
            foreach (var det in finalDetections)
            {
                var box = det.Box;
                var rect = new RectangleF(box[0] / inputWidth * drawImage.Width, box[1] / inputHeight * drawImage.Height,
                                          (box[2] - box[0]) / inputWidth * drawImage.Width,
                                          (box[3] - box[1]) / inputHeight * drawImage.Height);

                drawImage.Mutate(ctx =>
                {
                    // Draws red rectangle (polygon)
                    ctx.DrawPolygon(Color.Red, 3, new PointF[]
                    {
                        new(rect.Left, rect.Top),
                        new(rect.Right, rect.Top),
                        new(rect.Right, rect.Bottom),
                        new(rect.Left, rect.Bottom),
                        new(rect.Left, rect.Top)
                    });
                    ctx.DrawText($"{det.Label} {det.Confidence:P1}", font, Color.Yellow, new PointF(rect.Left, rect.Top - 25));
                });
            }

            imageWithBoxes = drawImage.Clone();
            return finalDetections;
        }

        private List<DetectionInput> ParseDetections(float[] output, float confThreshold)
        {
            var detections = new List<DetectionInput>();
            int numDetections = output.Length / (5 + _numClasses);

            for (int i = 0; i < numDetections; i++)
            {
                var offset = i * (5 + _numClasses);
                float cx = output[offset];
                float cy = output[offset + 1];
                float w = output[offset + 2];
                float h = output[offset + 3];
                float objConf = output[offset + 4];

                var classScores = output.Skip(offset + 5).Take(_numClasses).ToArray();
                var maxClassScore = classScores.Max();
                var labelIdx = Array.IndexOf(classScores, maxClassScore);

                float conf = objConf * maxClassScore;
                if (conf < confThreshold)
                    continue;

                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                detections.Add(new DetectionInput
                {
                    Label = _labels[labelIdx],
                    Confidence = conf,
                    Box = new[] { x1, y1, x2, y2 }
                });
            }

            return detections;
        }

        private List<DetectionInput> ApplyNms(List<DetectionInput> detections, float iouThreshold)
        {
            var finalDetections = new List<DetectionInput>();
            var grouped = detections.GroupBy(d => d.Label);

            foreach (var group in grouped)
            {
                var dets = group.OrderByDescending(d => d.Confidence).ToList();

                while (dets.Count > 0)
                {
                    var best = dets[0];
                    finalDetections.Add(best);
                    dets.RemoveAt(0);

                    dets = dets.Where(d => IoU(best.Box, d.Box) < iouThreshold).ToList();
                }
            }

            return finalDetections;
        }

        private float IoU(float[] boxA, float[] boxB)
        {
            float xA = Math.Max(boxA[0], boxB[0]);
            float yA = Math.Max(boxA[1], boxB[1]);
            float xB = Math.Min(boxA[2], boxB[2]);
            float yB = Math.Min(boxA[3], boxB[3]);

            float interArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
            float boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
            float boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

            return interArea / (boxAArea + boxBArea - interArea + 1e-6f);
        }
    }
}

import cv2

from models.mediapipe import MediapipeHandDetector
from models.yolo import YOLOHandDetector
from utils.dataset_lsc70_utils import load_reference_signs_lsc70
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


class MediapipeHolisticResults:
    def __init__(self, hand_results):
        self.left_hand_landmarks = hand_results.get("left_hand")
        self.right_hand_landmarks = hand_results.get("right_hand")
        self.pose_landmarks = None


if __name__ == "__main__":
    try:
        print("Cargando dataset LSC70...")
        reference_signs = load_reference_signs_lsc70()

        sign_recorder = SignRecorder(
            reference_signs,
            min_frames=10,
            max_frames=60,
            pause_threshold_frames=8,
            confidence_threshold=0.6,
            cooldown_seconds=0.8,
        )
        webcam_manager = WebcamManager()

        print("\nBuscando webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("ERROR: No se encontro webcam.")
            exit(1)

        print("Webcam lista!")
        print("Senales: HOLA, NOMBRE, BUENAS, TARDES, NOCHES, DIAS, GUSTAR, LICOR, YO, ANNOS")
        print("Controles: q=salir  c=borrar frase  v=cambiar vista (mediapipe/yolo/both)")

        hand_detector = MediapipeHandDetector(
            model_path="mediapipe/models/hand_landmarker.task",
            num_hands=2,
            static_image_mode=True,
        )

        try:
            yolo_detector = YOLOHandDetector(model_name="yolov8n.pt")
            print("YOLO cargado (usando yolov8n.pt)")
        except Exception as e:
            print(f"YOLO no disponible: {e}")
            yolo_detector = None

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Error leyendo frame de webcam")
                    break

                hand_results = hand_detector.detect(frame)
                results = MediapipeHolisticResults(hand_results)

                yolo_results = []
                if yolo_detector:
                    try:
                        yolo_results = yolo_detector.detect(frame)
                    except Exception:
                        pass

                sign_detected, is_recording = sign_recorder.process_results(results)

                confidence = sign_recorder.current_confidence if sign_detected else 0.0
                candidate = (
                    sign_recorder.candidate_sign
                    if not sign_detected and sign_recorder.candidate_sign
                    else ""
                )
                candidate_conf = sign_recorder.candidate_confidence if candidate else 0.0

                webcam_manager.update(
                    frame,
                    results,
                    yolo_results,
                    sign_detected,
                    is_recording,
                    sign_recorder.get_phrase(),
                    confidence,
                    candidate,
                    candidate_conf,
                )

                pressedKey = cv2.waitKey(1) & 0xFF
                if pressedKey == ord("q"):
                    print("Saliendo...")
                    break
                elif pressedKey == ord("c"):
                    sign_recorder.clear_phrase()
                    print("Frase borrada")
                elif pressedKey == ord("v"):
                    webcam_manager.toggle_view()
                    print(f"Vista: {webcam_manager.view_mode}")

            except KeyboardInterrupt:
                print("\nInterrumpido por usuario")
                break
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        print("Programa terminado.")

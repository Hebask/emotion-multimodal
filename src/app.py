import argparse
import json

from .utils.io import pretty_result

def main():
    parser = argparse.ArgumentParser(description="Emotion Multimodal Predictor")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--image", type=str, help="Path to image (face)")
    parser.add_argument("--audio", type=str, help="Path to audio WAV (speech)")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")

    args = parser.parse_args()

    if not (args.text or args.image or args.audio):
        parser.error("Provide at least one of --text, --image, --audio")

    results = []

    if args.text:
        from .text_emotion import predict_text_emotion
        r = predict_text_emotion(args.text)
        results.append(r)
        if not args.json:
            print(pretty_result("text", r["emotion"], r["score"]))

    if args.image:
        from .image_emotion import predict_image_emotion
        r = predict_image_emotion(args.image)
        results.append(r)
        if not args.json:
            print(pretty_result("image", r["emotion"], r["score"]))


    if args.audio:
        from .audio_emotion import predict_audio_emotion
        r = predict_audio_emotion(args.audio)
        results.append(r)
        if not args.json:
            print(pretty_result("audio", r["emotion"], r["score"]))


    if args.json:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

from jpeg_extractor import run_extraction

def main():
    input_path = "../../data/ts/input/holt.mp4"  
    output_path = "../../data/ts/output/holt"  
    run_extraction(input_path, output_path, fps=30)

if __name__ == "__main__":
    main()

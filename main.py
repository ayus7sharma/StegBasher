import os
import re
import argparse
import magic
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import subprocess
import string
from rich import print
from rich.table import Table
from rich.text import Text
from rich.markup import escape
import time

def display_table(title, content):
    table = Table(title=title)
    table.add_column("Information", style="bold cyan")
    
    if isinstance(content, list):  #added debuger for unintended dump content
        for line in content:
            table.add_row(line)
    else:
        for line in content.split('\n'):
            if line.strip():
                table.add_row(line)
    print(table)

def chi_square(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "[red]Could not load image.[/red]"
        
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        expected = np.ones(256) * np.mean(hist)
        chi_square = np.sum((hist - expected) ** 2 / expected)
        return f"[bold green]Chi-square statistic:[/bold green] {chi_square:.2f} (Higher value may indicate hidden data)"
    except Exception as e:
        return f"[red]Error detecting chi-square anomalies:[/red] {e}"


def parity_bit_steg(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "[red]Could not load image.[/red]"
        
        lsb = img & 1
        parity = np.sum(lsb) % 2
        return f"[bold green]Detected LSB parity:[/bold green] {parity} (Irregular parity may indicate hidden data)"
    except Exception as e:
        return f"[red]Error detecting parity anomalies:[/red] {e}"

def quant_table_extraction(image_path, all=False):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        is_success, enc_image = cv2.imencode(".jpg", image, encode_param)
        
        if not is_success:
            return "[red]Failed to extract quantization table.[/red]"
        
        jpeg = cv2.imdecode(enc_image, cv2.IMREAD_GRAYSCALE)
        quantization_table = jpeg[0:8, 0:8]  #8x8 matrix
        
        table_str = '\n'.join([' '.join(map(str, row)) for row in quantization_table])
        
        if all:
            return table_str
        else:
            with open("Quantization_Table.txt", "w") as f:
                f.write(table_str)
            return f"[bold green]Quantization table extracted and saved to:[/bold green] [Quantization_Table.txt] \n{table_str}"
    except Exception as e:
        return f"[red]Error extracting quantization table:[/red] {e}"

def dct_anomalies(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dct = cv2.dct(np.float32(image))
        high_freq_sum = np.sum(dct[8:, 8:])
        anomalies = np.sum(dct > 200)
        
        return (f"[bold green]DCT coefficient anomalies detected:[/bold green] {anomalies} high values\n"
                f"[bold green]High-frequency DCT coefficient sum:[/bold green] {high_freq_sum:.2f}")
    except Exception as e:
        return f"[red]Error detecting DCT anomalies:[/red] {e}"

def noice_analysis(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        noise = cv2.Laplacian(image, cv2.CV_64F).var()
        return f"[bold green]Image Noise Level:[/bold green] {noise:.2f}"
    except Exception as e:
        return f"[red]Error analyzing noise levels:[/red] {e}"

def histogram_anomaly_analysis(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        anomalies = np.sum(hist < 10)
        return f"[bold green]Histogram anomalies detected[/bold green]: {anomalies} bins with low frequency"
    except Exception as e:
        return f"[red]Error analyzing noise levels:[/red] {e}"

def metadata_extraction(image_path):
    try:
        result = subprocess.run(["exiftool", image_path], capture_output=True, text=True)
        return result.stdout if result.stdout else "[red]No metadata found.[/red]"
    except Exception as e:
        return f"[red]Error extracting metadata:[/red] {e}"

def palette_steg(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == "P":  #colmode
            return "[bold green]Indexed color mode detected. Possible palette-based steganography.[/bold green]"
        return "[red]No indexed color anomalies found.[/red]"
    except Exception as e:
        return f"[red]Error detecting palette anomalies:[/red] {e}"

def jsteg_jphide_detection(image_path):
    try:
        result = subprocess.run(["stegdetect", image_path], capture_output=True, text=True)
        return result.stdout if result.stdout else "[red]No JSteg/JPHide detected.[/red]" 
    except FileNotFoundError:
        return "[red]Stegdetect not found. Install it via 'sudo apt install stegdetect' | NOTE: This feature works only on Windows 7,10,11.[/red]"
    except Exception as e:
        return f"[red]Error detecting JSteg/JPHide:[/red] {e}"

def hidden_file_detection(image_path):
    try:
        with open(image_path, "rb") as f:
            raw_data = f.read()
        
        file_type = magic.Magic()
        detected_type = file_type.from_buffer(raw_data)

        return f"[bold green]Detected file type:[/bold green] {detected_type}"
    except Exception as e:
        return f"[red]Error detecting hidden files:[/red] {e}"

def extract_strings(image_path, min_length=4): #debugging length, can be modified.
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        extracted_strings = []
        current_string = ""
        
        for byte in data:
            if chr(byte) in string.printable:
                current_string += chr(byte)
                if len(current_string) >= min_length:
                    extracted_strings.append(current_string)
            else:
                current_string = ""
        
        return extracted_strings if extracted_strings else "[red]No readable strings found.[/red]"
    except Exception as e:
        return f"[red]Error extracting strings:[/red] {e}"


"""
def analyze_rgba_channels(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return "Could not load image."
        
        channels = cv2.split(image)
        channel_names = ["Blue", "Green", "Red", "Alpha"][:len(channels)]
        
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            cv2.imwrite(f"{name}_channel.png", channel)
        
        return f"Saved {', '.join(channel_names)} channels as separate images."
    except Exception as e:
        return f"Error analyzing RGBA channels: {e}"
"""

def phase_steg(image_path, extraction_dir=None):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "[red]Could not load image.[/red]"
        
        f_transform = np.fft.fft2(img)  #fft
        magnitude, phase = np.abs(f_transform), np.angle(f_transform)
        phase_image = np.exp(1j * phase)
        reconstructed = np.fft.ifft2(phase_image).real
        
        if extraction_dir:
            phase_dir = os.path.join(extraction_dir, "Phase")
            os.makedirs(phase_dir, exist_ok=True)
            output_path = os.path.join(phase_dir, "Analysed_Phase.png")
        else:
            output_path = "Analysed_Phase.png"
        
        cv2.imwrite(output_path, reconstructed)
        return f"[bold green]Phase-based analysis image saved as:[/bold green] '{output_path}'"
    except Exception as e:
        return f"[red]Error detecting phase anomalies:[/red] {e}"

def alpha_channel_analysis(image_path, extraction_dir=None):
    try:
        if extraction_dir:
            alpha_dir = os.path.join(extraction_dir, "Alpha Channel")
            os.makedirs(alpha_dir, exist_ok=True)
            output_path = os.path.join(alpha_dir, "Alpha_Channel.png")
        else:
            output_path = "Alpha_Channel.png"

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return "[red]Could not load image.[/red]"
    
        if image.shape[-1] < 4:
            if extraction_dir:
                placeholder_path = os.path.join(alpha_dir, "Alpha_Channel_Analysis.txt")
                with open(placeholder_path, "w") as f:
                    f.write("No alpha channel was found in the image.\n")
            return "[red]No alpha channel found.[/red]"

        alpha_channel = image[:, :, 3]
        
        cv2.imwrite(output_path, alpha_channel)
        return f"[bold green]Alpha channel extracted and saved as:[/bold green] '{output_path}'"
    except Exception as e:
        return f"[red]Error analyzing alpha channel:[/red] {e}"

def eof_extraction(image_path, extraction_dir=None):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        
        eof_marker = b"\xFF\xD9"  #raw
        if eof_marker in data:
            end_offset = data.index(eof_marker) + 2
            hidden_data = data[end_offset:]
            
            if hidden_data:
                if extraction_dir:
                    eof_dir = os.path.join(extraction_dir, "EOF")
                    os.makedirs(eof_dir, exist_ok=True)
                    output_path = os.path.join(eof_dir, "data.bin")
                else:
                    output_path = "data.bin"
                
                with open(output_path, "wb") as f:
                    f.write(hidden_data)
                return f"[bold green]Hidden EOF data detected and saved as:[/bold green] '{output_path}'"
        
        return "[red]No hidden EOF data detected.[/red]"
    except Exception as e:
        return f"[red]Error detecting EOF data:[/red] {e}"

def color_plane_extraction(image_path, extraction_dir=None):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "[red]Could not load image.[/red]"

        channels = cv2.split(image)
        color_names = ["Blue", "Green", "Red"]
        
        if extraction_dir:
            color_dir = os.path.join(extraction_dir, "Colour Planes")
            os.makedirs(color_dir, exist_ok=True)
        else:
            color_dir = "."
        
        for i, (channel, name) in enumerate(zip(channels, color_names)):
            output_path = os.path.join(color_dir, f"{name}_Plane.png")
            cv2.imwrite(output_path, channel)
        
        return f"[bold green]Saved {', '.join(color_names)} color planes as separate images in '{color_dir}'.[/bold green]"
    except Exception as e:
        return f"[bold green]Error extracting color planes:[/bold green] {e}"

def main():
    p = argparse.ArgumentParser(description="StegBasher v0.1 - A Vast CLI-Based Image Steganography Tool")
    p.add_argument("image", help="Path to the image file")

    p.add_argument("-m", "--metadata", action="store_true", help="Extract metadata (EXIF)")
    p.add_argument("-s", "--strings", action="store_true", help="Extract readable strings")
    p.add_argument("-p", "--palette", action="store_true", help="Detect palette-based steg")
    p.add_argument("-c", "--planes", action="store_true", help="Extract RGB color planes")
    p.add_argument("-a", "--alpha", action="store_true", help="Analyze alpha channel")
    p.add_argument("-e", "--eof", action="store_true", help="Check for hidden EOF data")
    p.add_argument("-q", "--quantization", action="store_true", help="Extract JPEG quantization table")
    p.add_argument("-n", "--noise", action="store_true", help="Analyze image noise levels")
    p.add_argument("-j", "--jsteg", action="store_true", help="Detect JSteg/JPHide in JPEGs") #win7,10,11 only
    p.add_argument("-chi", "--chi-square", action="store_true", help="Detect chi-square anomalies")
    p.add_argument("-pha", "--phase", action="store_true", help="Detect phase-based steganography")
    p.add_argument("-par", "--parity", action="store_true", help="Analyze parity bit patterns")
    p.add_argument("-dct", "--dct", action="store_true", help="Detect hidden data in DCT coefficients")
    p.add_argument("-all","--allsteg", action="store_true", help="Applies all of the above Steganographic features.")
    p.add_argument("-hist", "--histogram", action="store_true", help="Detect histogram anomalies")

    args = p.parse_args()

    if args.strings:
        print(f"\n[bold yellow]=== Strings Extraction | {args.image} ===[/bold yellow]")
        strings = extract_strings(args.image)
        print("\n[bold blue]Results[/bold blue]", strings)

    if args.metadata:
        print(f"\n[bold yellow]=== Metadata Extraction | {args.image} ===[/bold yellow]")
        metadata = metadata_extraction(args.image)
        display_table("[bold blue]Results[/bold blue]", metadata)

    if args.palette:
        print(f"\n[bold yellow]=== Palette-Based Steganography Detection | {args.image} ===[/bold yellow]")
        print(palette_steg(args.image))

    if args.alpha:
        print(f"\n[bold yellow]=== Alpha Channel Analysis | {args.image} ===[/bold yellow]")
        print(alpha_channel_analysis(args.image))

    if args.eof:
        print(f"\n[bold yellow]=== End-of-File (EOF) Data Extraction | {args.image} ===[/bold yellow]")
        print(eof_extraction(args.image))

    if args.planes:
        print(f"\n[bold yellow]=== StegSolve Color Planes Extraction | {args.image} ===[/bold yellow]")
        print(color_plane_extraction(args.image))

    if args.jsteg:
        print(f"\n[bold yellow]=== JSteg/JPHide Detection | {args.image} ===[/bold yellow]")
        print(jsteg_jphide_detection(args.image))

    if args.chi_square:
        print(f"\n[bold yellow]=== Chi-Square Analysis | {args.image} ===[/bold yellow]")
        print(chi_square(args.image))
    
    if args.phase:
        print(f"\n[bold yellow]=== Phase-Based Steganography Detection | {args.image} ===[/bold yellow]")
        print(phase_steg(args.image))
    
    if args.parity:
        print(f"\n[bold yellow]=== Parity Bit Analysis | {args.image} ===[/bold yellow]")
        print(parity_bit_steg(args.image))
    
    if args.dct:
        print(f"\n[bold yellow]=== DCT Coefficient Analysis | {args.image} ===[/bold yellow]")
        print(dct_anomalies(args.image))

    if args.noise:
        print(f"\n[bold yellow]=== Noise Level Analysis | {args.image} ===[/bold yellow]")
        print(noice_analysis(args.image))

    if args.histogram:
        print(f"\n[bold yellow]=== Histogram Anomaly Detection | {args.image} ===[/bold yellow]")
        print(histogram_anomaly_analysis(args.image))

    if args.quantization:
        print(f"\n[bold yellow]=== Quantization Table Extraction | {args.image} ===[/bold yellow]")
        quantization_table = quant_table_extraction(args.image)
        display_table("[bold blue]Results[/bold blue]", quantization_table)


    if args.allsteg:
        print(f"\n[bold yellow]=== Complete Analysis | {args.image} ===[/bold yellow]")
        time.sleep(1)
        print(f"\n[bold magenta]NOTE: This feature is currently in beta testing, please report any bugs you find. [/bold magenta]")
        time.sleep(1)

        extraction_dir = "extractions"
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)

        def save_plain_txt(file_path, content): #stripping raw rich.text shit
            plain_text = re.sub(r"\[.*?\]", "", content)
            with open(file_path, "w") as f:
                f.write(plain_text)

        #str
        print(f"\n[bold green]=== Strings Extraction | [1/14] ===[/bold green]")
        strings = extract_strings(args.image)
        with open(f"{extraction_dir}/Strings_{args.image}.txt", "w") as s:
            s.write(str(strings))
        print(f"[bold green]Strings Extraction Complete ✅ | Saved as {extraction_dir}/Strings_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #meta
        print(f"\n[bold green]=== Metadata Extraction | [2/14] ===[/bold green]")
        metadata = metadata_extraction(args.image)
        save_plain_txt(f"{extraction_dir}/Metadata_{args.image}.txt", metadata)
        print(f"[bold green]Metadata Extraction Complete ✅ | Saved as {extraction_dir}/Metadata_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #palette
        print(f"\n[bold green]=== Palette-Based Steganography Detection | [3/14] ===[/bold green]")
        palette_result = palette_steg(args.image)
        save_plain_txt(f"{extraction_dir}/Palette_Steg_{args.image}.txt", palette_result)
        print(f"[bold green]Palette-Based Steganography Detection Complete ✅ | Saved as {extraction_dir}/Palette_Steg_{args.image}.txt[/bold green]")
        
        time.sleep(0.5)

        #alpha (more like sigma)
        print(f"\n[bold green]=== Alpha Channel Analysis | [4/14] ===[/bold green]")
        alpha_result = alpha_channel_analysis(args.image, extraction_dir)
        save_plain_txt(f"{extraction_dir}/Alpha_Channel_{args.image}.txt", alpha_result)
        print(f"[bold green]Alpha Channel Analysis Complete ✅ | Saved as {extraction_dir}/Alpha_Channel_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #eof
        print(f"\n[bold green]=== End-of-File (EOF) Data Extraction | [5/14] ===[/bold green]")
        eof_result = eof_extraction(args.image, extraction_dir)
        save_plain_txt(f"{extraction_dir}/EOF_Data_{args.image}.txt", eof_result)
        print(f"[bold green]EOF Data Extraction Complete ✅ | Saved as {extraction_dir}/EOF_Data_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #rgb
        print(f"\n[bold green]=== RGB Color Planes Extraction | [6/14] ===[/bold green]")
        planes_result = color_plane_extraction(args.image, extraction_dir)
        save_plain_txt(f"{extraction_dir}/Color_Planes_{args.image}.txt", planes_result)
        print(f"[bold green]RGB Color Planes Extraction Complete ✅ | Saved as {extraction_dir}/Color_Planes_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #jsteg/jphide
        print(f"\n[bold green]=== JSteg/JPHide Detection | [7/14] ===[/bold green]")
        jsteg_result = jsteg_jphide_detection(args.image)
        save_plain_txt(f"{extraction_dir}/JSteg_JPHide_{args.image}.txt", jsteg_result)
        print(f"[bold green]JSteg/JPHide Detection Complete ✅ | Saved as {extraction_dir}/JSteg_JPHide_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #chi-sq
        print(f"\n[bold green]=== Chi-Square Analysis | [8/14] ===[/bold green]")
        chi_square_result = chi_square(args.image)
        save_plain_txt(f"{extraction_dir}/Chi_Square_{args.image}.txt", chi_square_result)
        print(f"[bold green]Chi-Square Analysis Complete ✅ | Saved as {extraction_dir}/Chi_Square_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #phase
        print(f"\n[bold green]=== Phase-Based Steganography Detection | [9/14] ===[/bold green]")
        phase_result = phase_steg(args.image, extraction_dir)
        save_plain_txt(f"{extraction_dir}/Phase_Steg_{args.image}.txt", phase_result)
        print(f"[bold green]Phase-Based Steganography Detection Complete ✅ | Saved as {extraction_dir}/Phase_Steg_{args.image}.txt[/bold green]")
        time.sleep(0.5)

        #parity
        print(f"\n[bold green]=== Parity Bit Analysis | [10/14] ===[/bold green]")
        parity_result = parity_bit_steg(args.image)
        save_plain_txt(f"{extraction_dir}/Parity_Bit_{args.image}.txt", parity_result)
        print(f"[bold green]Parity Bit Analysis Complete ✅ | Saved as {extraction_dir}/Parity_Bit_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #dct
        print(f"\n[bold green]=== DCT Coefficient Analysis | [11/14] ===[/bold green]")
        dct_result = dct_anomalies(args.image)
        save_plain_txt(f"{extraction_dir}/DCT_Coefficients_{args.image}.txt", dct_result)
        print(f"[bold green]DCT Coefficient Analysis Complete ✅ | Saved as {extraction_dir}/DCT_Coefficients_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        # noise
        print(f"\n[bold green]=== Noise Level Analysis | [12/14] ===[/bold green]")
        noise_result = noice_analysis(args.image)
        save_plain_txt(f"{extraction_dir}/Noise_Level_{args.image}.txt", noise_result)
        print(f"[bold green]Noise Level Analysis Complete ✅ | Saved as {extraction_dir}/Noise_Level_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #histo
        print(f"\n[bold green]=== Histogram Anomaly Detection | [13/14] ===[/bold green]")
        histogram_result = histogram_anomaly_analysis(args.image)
        save_plain_txt(f"{extraction_dir}/Histogram_Anomalies_{args.image}.txt", histogram_result)
        print(f"[bold green]Histogram Anomaly Detection Complete ✅ | Saved as {extraction_dir}/Histogram_Anomalies_{args.image}.txt[/bold green]")

        time.sleep(0.5)

        #quant
        print(f"\n[bold green]=== Quantization Table Extraction | [14/14] ===[/bold green]")
        quantization_result = quant_table_extraction(args.image, all=True)
        with open(f"{extraction_dir}/Quantization_Table_{args.image}.txt", "w") as q:
            q.write(quantization_result)
        print(f"[bold green]Quantization Table Extraction Complete ✅ | Saved as {extraction_dir}/Quantization_Table_{args.image}.txt[/bold green]")

        time.sleep(1)
        print(f"\n[bold yellow]=== All Steganographic Features Applied Successfully ✅ ===[/bold yellow]")

if __name__ == "__main__":
    main()

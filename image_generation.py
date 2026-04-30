import os
import random
from PIL import Image
import concurrent.futures
from tqdm.auto import tqdm

# --- PFADE (Bitte anpassen) ---
FOREGROUND_DIR = './data/asl_alphabet_train_nobg'
BACKGROUND_DIR = './data/val2017'
OUTPUT_DIR = './data/offline_composited'
VERSIONS_PER_IMAGE = 2  # 2 Hintergründe pro Hand = 174.000 Bilder

# Sammle alle Hintergrund-Pfade vorab
bg_paths = [os.path.join(BACKGROUND_DIR, img) for img in os.listdir(BACKGROUND_DIR)
            if img.lower().endswith(('.jpg', '.png', '.jpeg'))]


def process_single_image(args):
    """Worker-Funktion für einen einzelnen CPU-Thread."""
    fg_path, cls_name, img_name = args

    try:
        # Lade die freigestellte Hand (RGBA)
        fg_img = Image.open(fg_path).convert("RGBA")
        fg_width, fg_height = fg_img.size

        for version in range(VERSIONS_PER_IMAGE):
            # Zufälligen Hintergrund laden und auf Hand-Größe zuschneiden
            bg_path = random.choice(bg_paths)
            bg_img = Image.open(bg_path).convert("RGB")

            # Aspekt-Ratio beibehalten, aber auf exakte Größe zuschneiden
            bg_img = bg_img.resize((fg_width, fg_height))
            bg_img = bg_img.convert("RGBA")

            # Alpha-Compositing
            composited = Image.alpha_composite(bg_img, fg_img).convert("RGB")

            # Speichern als fertiges, leichtes JPEG
            out_name = f"{os.path.splitext(img_name)[0]}_v{version}.jpg"
            out_path = os.path.join(OUTPUT_DIR, cls_name, out_name)
            composited.save(out_path, format="JPEG", quality=90)

        return True
    except Exception as e:
        # Falls ein Bild defekt ist
        return False


def main():
    print(f"Starte High-Speed Compositing auf CPU...")

    # Zielordner-Struktur (Klassen A-Z) aufbauen
    classes = sorted(os.listdir(FOREGROUND_DIR))
    tasks = []

    for cls_name in classes:
        cls_dir = os.path.join(FOREGROUND_DIR, cls_name)
        if os.path.isdir(cls_dir):
            # Erstelle den passenden Unterordner im Output-Verzeichnis
            os.makedirs(os.path.join(OUTPUT_DIR, cls_name), exist_ok=True)

            # Sammle alle Aufgaben
            for img_name in os.listdir(cls_dir):
                fg_path = os.path.join(cls_dir, img_name)
                tasks.append((fg_path, cls_name, img_name))

    # Nutze ALLE Threads deines Ryzen 8700X3D
    total_images = len(tasks) * VERSIONS_PER_IMAGE
    print(f"Generiere insgesamt {total_images} fertige Trainingsbilder...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Verarbeite Originalbilder"))

    print("\n✅ Compositing abgeschlossen! Dein Dataset ist bereit für die GPU.")


if __name__ == '__main__':
    main()
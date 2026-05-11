# prueba/generar_mapa_csv.py
import csv
from pathlib import Path
import folium

CSV_PATH = Path(__file__).parent / "coordenadas.csv"
OUTPUT_HTML = Path(__file__).parent / "mapa_citricos_csv.html"

def parse_float(x):
    if x is None: 
        return None
    x = x.strip()
    if not x:
        return None
    # cambia coma por punto si aparece
    x = x.replace(",", ".")
    return float(x)

def main():
    print("### generar_mapa_csv.py ###")
    print(f"CSV: {CSV_PATH}")

    if not CSV_PATH.exists():
        print("❌ No existe coordenadas.csv. Crea el CSV primero.")
        return

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            nombre = row.get("archivo")
            lat = parse_float(row.get("latitud"))
            lon = parse_float(row.get("longitud"))
            alt = parse_float(row.get("altitud")) if "altitud" in row else None
            if nombre and lat is not None and lon is not None:
                rows.append((nombre, lat, lon, alt))
            else:
                print(f"   – Línea {i}: datos incompletos, se ignora.")

    if not rows:
        print("⚠️  No hay filas válidas en el CSV.")
        return

    # Centrar el mapa en el primer punto
    m = folium.Map(location=[rows[0][1], rows[0][2]], zoom_start=17)

    for nombre, lat, lon, alt in rows:
        html = f"<b>{nombre}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        if alt is not None:
            html += f"<br>Alt: {alt:.2f} m"
        folium.Marker(
            location=[lat, lon],
            popup=html,
            icon=folium.Icon(color="green", icon="ok-sign")
        ).add_to(m)

    m.save(str(OUTPUT_HTML))
    print(f"✅ Mapa guardado en: {OUTPUT_HTML}")
    print("   Ábrelo con: start " + str(OUTPUT_HTML))

if __name__ == "__main__":
    main()

#".\.venv\Scripts\Activate.ps1"
#python ".\prueba\generar_mapa_csv.py"
#start ".\prueba\mapa_citricos_csv.html"

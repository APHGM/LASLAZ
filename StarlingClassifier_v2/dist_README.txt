Starling Ground-Contact Classifier
===================================

WHAT THIS IS
------------
A standalone Windows application that classifies ground and starling
bird-contact points in terrestrial LiDAR scans (.laz / .las files).
No installation required — no Python needed.

HOW TO RUN
----------
1. Extract this ZIP to any folder on your computer (e.g. C:\Apps\StarlingClassifier\).
   IMPORTANT: Extract the ENTIRE folder, not just the .exe file.
   The .exe needs the files alongside it to run.

2. Open the extracted folder.

3. Double-click  StarlingClassifier.exe

4. In the GUI:
   - Pick "Single LAZ file" mode (or "Tile folder" if you already have tiles)
   - Browse to your LAZ file
   - Choose an output folder (anywhere on a local drive)
   - Set tile size (100 m default, use 25 m for very dense scans)
   - Click "Start Processing"

OUTPUT
------
The app produces:
  - <tile_name>_classified.laz   (LAZ with classification codes:
                                   1 = unclassified, 2 = ground, 20 = bird contact)
  - bird_contacts.csv            (centroid X/Y/Z + cluster size for each bird)

VIEW RESULTS
------------
Open the classified .laz files in:
  - CloudCompare (free)
  - Trimble RealWorks / Leica Cyclone
  - QGIS with LAStools

In CloudCompare, colour by "Classification" scalar field to see ground
(brown), birds (yellow/orange), and unclassified (default).

WINDOWS DEFENDER / ANTIVIRUS
----------------------------
Some antivirus tools flag PyInstaller-built apps as "unknown". This is a
common false-positive — the app is safe. If blocked, ask IT to whitelist
the folder, or run from a folder that isn't restricted.

TROUBLESHOOTING
---------------
"The application can't start because ... is missing":
   → You extracted only the .exe, not the whole folder. Re-extract.

GUI doesn't open / crashes immediately:
   → Try running from a Command Prompt to see the error:
     cd \path\to\extracted\folder
     StarlingClassifier.exe

"LAZ write failed":
   → Choose Output Format = LAS (uncompressed) in the GUI.

SUPPORT
-------
Contact the person who sent you this build.

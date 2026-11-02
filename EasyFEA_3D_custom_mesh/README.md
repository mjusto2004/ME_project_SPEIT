# Maillage 3D personnalisé EasyFEA (version de remplacement de Thermal2)

Ce package est une version de remplacement de `examples/Thermal/Thermal2.py` utilisant un maillage 3D quelconque. Il conserve les paramètres d'origine de la conduction thermique transitoire en 3D (`alpha=0.5`, 20 pas, 0 à gauche / 40 à droite, adiabatique ailleurs) et remplace uniquement la génération automatique du maillage par gmsh à partir de la géométrie par l'import d'un maillage personnalisé fourni par l'utilisateur.

## 1. Contenu du package

| Fichier | Description |
| --- | --- |
| `thermal2_custom_mesh.py` | Programme principal en ligne de commande pour le calcul transitoire 3D, correspondant à la version modifiée de `Thermal2.py` |
| `Thermal2_custom.py` | Version avec les modifications minimales, ligne par ligne, du Thermal2 d'origine |
| `custom_mesh.py` | Cœur de l'import de maillages personnalisés (2D/3D) : `MeshData`, validation, correction de l'orientation, formats JSON/NPZ/CSV/MSH |
| `demo_meshes_3d.py` | Reproduit le maillage du prisme carré à trou circulaire de Thermal2 ; contient aussi un maillage HEXA8 fait à la main pour un prisme carré creux |
| `demo_meshes.py` | Dépendance partagée : fournit notamment `mesh_to_data()` |
| `thermal_custom_mesh.py` | Dépendance partagée : fournit `resolve_bc_nodes()`, `parse_bcs()` et `solve_thermal()` pour le régime permanent |
| `verify_custom_mesh_3d.py` | 40 vérifications automatisées en 3D |
| `env_compat.py` | Couche de compatibilité pour la couche de threads MKL de cette machine |
| `data/thermal2_TETRA4/` | Maillage d'exemple Thermal2 (JSON/NPZ/CSV + étiquettes) |
| `data/hollow_box_HEXA8/` | Maillage d'exemple fait à la main d'un prisme carré creux en 3D |
| `data/box3d_TETRA4.msh` | Exemple d'import d'un fichier `.msh` gmsh |
| `results/` | Fichiers CSV des résultats transitoires d'exemple |

## 2. Environnement

- Environnement Anaconda : `Fea`
- Python : `C:\Users\Administrator.DESKTOP-O3D0EMQ\anaconda3\envs\Fea\python.exe`
- EasyFEA 7.1.0 (nom d'import `EasyFEA`, et non `easyfea`)
- gmsh 4.15.2 (nécessaire uniquement pour l'import `.msh` et le maillage de référence Thermal2)

Sur cette machine, la couche de threads Intel/TBB utilisée par défaut avec MKL 2026.1.0 peut provoquer un plantage natif au premier appel à `np.matmul` ; `env_compat.py` définit `MKL_THREADING_LAYER=SEQUENTIAL` avant l'import de NumPy. Tous les scripts d'entrée l'importent en premier.

## 3. Démarrage rapide

Après extraction, placez-vous dans le répertoire racine du package (celui qui contient `thermal2_custom_mesh.py`) :

```powershell
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\Administrator.DESKTOP-O3D0EMQ\anaconda3\envs\Fea\python.exe"

# 1) Reproduire exactement la géométrie de Thermal2, résoudre le problème transitoire via l'interface personnalisée et comparer au maillage d'origine
python3 thermal2_custom_mesh.py --demo thermal2 --elem-type TETRA4

# 2) Prisme carré creux 3D entièrement défini à la main (HEXA8, sans gmsh)
python3 thermal2_custom_mesh.py --demo box --n 6 --nz 2

# 3) Charger le maillage JSON d'exemple fourni avec le package
python3 thermal2_custom_mesh.py --mesh data/thermal2_TETRA4/mesh.json

# 4) Charger un fichier .msh 3D de gmsh
python3 thermal2_custom_mesh.py --mesh data/box3d_TETRA4.msh

# 5) Version avec les modifications minimales, ligne par ligne
python3 Thermal2_custom.py

# 6) Vérification complète en 3D
python3 verify_custom_mesh_3d.py
```

Paramètres courants :

```text
--tmax 0.5 --steps 20 --alpha 0.5
--bc left:0 --bc right:40
--export data/my_mesh --save results/my_result.csv
```

`--save` écrit les colonnes `node,x,y,z,t` ; `--export` peut produire un fichier JSON/NPZ ou un répertoire contenant `nodes.csv`, `elements.csv` et `node_tags.json`.

### Utiliser un maillage personnalisé avec `Thermal2_Flux.py`

Depuis la racine du projet, le maillage peut être chargé directement par le script de flux :

```powershell
python Thermal2_Flux.py --mesh EasyFEA_3D_custom_mesh/data/thermal2_TETRA4/mesh.json --diameter 0.3333333333
```

Le script utilise l'étiquette `circle` pour la surface chauffée si elle existe ; sinon, il sélectionne les nœuds du cercle à partir du diamètre (`--diameter`, 0.25 par défaut) et du centre de la boîte englobante. Pour la convection, il utilise l'étiquette `boundary` si elle existe, en excluant les nœuds chauffés ; sinon, il sélectionne les plans extérieurs du maillage. Des étiquettes de convection spécifiques peuvent être données en répétant `--convection-tag`, et `--tol` règle la tolérance géométrique.

## 4. Éléments pris en charge et ordre des nœuds

Éléments pris en charge en 3D :

| Type | Nombre de nœuds | Ordre |
| --- | --- | --- |
| `TETRA4` | 4 | `v0, v1, v2, v3` |
| `TETRA10` | 10 | 4 sommets + 6 milieux d'arêtes (ordre gmsh) |
| `HEXA8` | 8 | Face inférieure `0..3`, face supérieure `4..7` |
| `HEXA20` | 20 | `HEXA8` + 12 milieux d'arêtes |
| `HEXA27` | 27 | `HEXA20` + nœuds de face et de volume |
| `PRISM6` | 6 | Triangle inférieur `0..2`, triangle supérieur `3..5` |
| `PRISM15` | 15 | `PRISM6` + 9 milieux d'arêtes |
| `PRISM18` | 18 | `PRISM15` + 3 nœuds de face |

- L'indexation est 0-based par défaut ; `--index-base 1` permet de lire des indices 1-based.
- Les éléments 3D doivent avoir un volume positif. L'inversion automatique est prise en charge pour `TETRA4 / HEXA8 / PRISM6` ; pour les éléments 3D d'ordre supérieur, le préprocesseur doit fournir une connectivité donnant un volume positif.
- Vérifiés numériquement : `TETRA4`, `TETRA10`, `HEXA8`, `PRISM6` et `PRISM18`.

## 5. Maillages et résultats d'exemple

Le package comprend :

- `data/thermal2_TETRA4/` : 240 nœuds, 864 éléments TETRA4, volume `0.921432579868`
- `data/hollow_box_HEXA8/` : 1008 nœuds, 576 éléments HEXA8, volume exact de 8
- `data/box3d_TETRA4.msh` : maillage tétraédrique gmsh de dimensions 2×1×1, volume 2
- `results/thermal2_TETRA4.csv`, `results/hollow_box_HEXA8.csv` : état final après 20 pas de calcul transitoire

Résultats des vérifications :

| Vérification | Résultat |
| --- | --- |
| Test de patch Thermal2 TETRA4 `T=2x+3y+4z` | `8.9e-16` |
| Résultat final transitoire Thermal2 TETRA4 par rapport à la version d'origine | Écart maximal entre nœuds : `0` |
| TETRA10 / PRISM18 | Écart du test de patch : `~5e-15`, écart transitoire : `0` |
| Prisme carré creux HEXA8 fait à la main | Écart du test de patch : `1.1e-14` |
| Import 3D `.msh` | Écart du test de patch : `2.7e-15` |
| `verify_custom_mesh_3d.py` | `40 / 40` réussies |

## 6. Limitations

- Un maillage 3D doit utiliser un seul type d'élément principal (ne mélangez pas des TETRA4 et des HEXA8 dans un même maillage).
- La correction automatique de l'orientation des éléments 3D d'ordre supérieur n'est pas implémentée ; vérifiez que leur connectivité donne un volume positif.
- Le script 3D actuel produit uniquement des fichiers CSV ; il ne génère ni cartes de contours ni animations 3D.
- L'import `.msh` dépend de gmsh ; l'utilisation de fichiers personnalisés JSON/NPZ/CSV ne nécessite pas gmsh.
- La tolérance par défaut pour la sélection des coordonnées est de `1e-9` fois la longueur de la diagonale du maillage ; elle peut être ajustée avec `--tol`.

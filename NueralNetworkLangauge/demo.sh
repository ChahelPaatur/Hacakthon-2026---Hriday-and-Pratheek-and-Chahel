#!/bin/bash
# ╔═══════════════════════════════════════════════╗
# ║  NeuroLang — Practice Demo Script             ║
# ║  Run: bash demo.sh                            ║
# ╚═══════════════════════════════════════════════╝

set -e
cd "$(dirname "$0")"

Y="\033[33m"
G="\033[32m"
B="\033[1m"
D="\033[2m"
X="\033[0m"

pause() {
  echo ""
  echo -e "${D}Press Enter to continue...${X}"
  read -r
  echo ""
}

echo -e "${Y}${B}"
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  NeuroLang Demo — Interactive Walkthrough     ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo -e "${X}"

# ─────────────────────────────────────────────────
echo -e "${B}1. COMPILE A SIMPLE PROGRAM${X}"
echo -e "${D}   This compiles iris.nl and shows the architecture summary.${X}"
pause
npx tsx src/cli.ts examples/iris.nl

# ─────────────────────────────────────────────────
echo -e "${B}2. GENERATE PYTORCH CODE${X}"
echo -e "${D}   Same program, but emit production-ready PyTorch Python code.${X}"
pause
npx tsx src/cli.ts examples/iris.nl --emit-code --target pytorch

# ─────────────────────────────────────────────────
echo -e "${B}3. GENERATE KERAS CODE${X}"
echo -e "${D}   Emit Keras Python code — ready to copy-paste into any ML project.${X}"
pause
npx tsx src/cli.ts examples/iris.nl --emit-code --target keras

# ─────────────────────────────────────────────────
echo -e "${B}4. GENERATE JAX/FLAX CODE${X}"
echo -e "${D}   Emit JAX + Flax code for high-performance research workflows.${X}"
pause
npx tsx src/cli.ts examples/wine.nl --emit-code --target jax

# ─────────────────────────────────────────────────
echo -e "${B}5. TRAIN A MODEL (Iris — 3-class classification)${X}"
echo -e "${D}   Actually train the neural network and see live results.${X}"
pause
npx tsx src/cli.ts examples/iris.nl --run

# ─────────────────────────────────────────────────
echo -e "${B}6. FULL-FEATURED TRAINING (with cross-validation)${X}"
echo -e "${D}   Deep network + batch norm + dropout + early stopping + 5-fold CV.${X}"
pause
npx tsx src/cli.ts examples/full-demo.nl --run

# ─────────────────────────────────────────────────
echo -e "${B}7. ENSEMBLE LEARNING${X}"
echo -e "${D}   Train 3 models and combine their predictions (majority vote).${X}"
pause
npx tsx src/cli.ts examples/ensemble.nl --run

# ─────────────────────────────────────────────────
echo -e "${B}8. FEATURE IMPORTANCE (Explainability)${X}"
echo -e "${D}   Which features matter most? Permutation importance tells you.${X}"
pause
npx tsx src/cli.ts examples/iris.nl --run --explain

# ─────────────────────────────────────────────────
echo -e "${B}9. BENCHMARK SUITE${X}"
echo -e "${D}   Run all 5 built-in datasets and see a results table.${X}"
echo -e "${D}   (This takes about 60 seconds)${X}"
pause
npx tsx src/cli.ts --benchmark

# ─────────────────────────────────────────────────
echo ""
echo -e "${G}${B}Demo complete!${X}"
echo ""
echo -e "${B}Try writing your own .nl programs:${X}"
echo ""
echo -e "  ${Y}task classification${X}"
echo -e "  ${Y}predict species${X}"
echo -e "  ${Y}inputs sepal_length sepal_width petal_length petal_width${X}"
echo -e "  ${Y}dataset iris.csv${X}"
echo -e "  ${Y}learn deep${X}"
echo -e "  ${Y}ensemble 5${X}"
echo -e "  ${Y}cross_validate 5${X}"
echo ""
echo -e "${D}Available datasets: iris, wine, titanic, digits, housing${X}"
echo -e "${D}Available targets:  tensorflow, pytorch, keras, jax, summary${X}"
echo -e "${D}Run: npx tsx src/cli.ts <yourfile.nl> --run${X}"
echo ""

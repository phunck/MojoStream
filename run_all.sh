#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.pixi/bin:$PATH"

PY_OUT="$(mktemp)"
MJ_OUT="$(mktemp)"
trap 'rm -f "$PY_OUT" "$MJ_OUT"' EXIT

echo "############################################"
echo "#  Lauf 1: Python / NumPy                   "
echo "############################################"
python3 matrix_bench.py | tee "$PY_OUT"

echo
echo "############################################"
echo "#  Lauf 2: Mojo (SIMD + parallelize + Tiling)"
echo "############################################"
pixi run mojo build matrix_bench.mojo -o matrix_bench
./matrix_bench | tee "$MJ_OUT"

get() { grep -E "^$1=" "$2" | tail -n1 | cut -d= -f2 | tr -d ' '; }

NP_MS=$(get RESULT_NUMPY_MS         "$PY_OUT")
NP_GF=$(get RESULT_NUMPY_GFLOPS     "$PY_OUT")
PY_MS=$(get RESULT_PYTHON_MS        "$PY_OUT")
PY_GF=$(get RESULT_PYTHON_GFLOPS    "$PY_OUT")
MJ_MS=$(get RESULT_MOJO_MS          "$MJ_OUT")
MJ_GF=$(get RESULT_MOJO_GFLOPS      "$MJ_OUT")
MT_MS=$(get RESULT_MOJO_TILED_MS    "$MJ_OUT")
MT_GF=$(get RESULT_MOJO_TILED_GFLOPS "$MJ_OUT")

calc() { python3 -c "print(f'{($1):.2f}')"; }

SP_PY_VS_MT=$(calc "$PY_MS / $MT_MS")
SP_NP_VS_MT=$(calc "$NP_MS / $MT_MS")
SP_MJ_VS_MT=$(calc "$MJ_MS / $MT_MS")
SP_MT_VS_MJ=$(calc "$MJ_MS / $MT_MS")

echo
echo "######################################################################"
echo "#               BENCHMARK-ZUSAMMENFASSUNG  (N=1024 fp32)            #"
echo "######################################################################"
printf "%-20s | %12s | %12s | %18s\n" "Backend" "Zeit [ms]" "GFLOPS" "Speedup vs Tiled"
printf -- "---------------------+--------------+--------------+--------------------\n"
printf "%-20s | %12s | %12s | %15s x\n" "Pure Python*"     "$PY_MS" "$PY_GF" "$SP_PY_VS_MT"
printf "%-20s | %12s | %12s | %15s x\n" "NumPy (BLAS)"     "$NP_MS" "$NP_GF" "$SP_NP_VS_MT"
printf "%-20s | %12s | %12s | %15s x\n" "Mojo SIMD+par"    "$MJ_MS" "$MJ_GF" "$SP_MJ_VS_MT"
printf "%-20s | %12s | %12s | %15s x\n" "Mojo Tiled+SIMD"  "$MT_MS" "$MT_GF" "1.00"
echo
echo "Tiling-Gewinn innerhalb von Mojo (Tiled vs. Naiv-SIMD): ${SP_MT_VS_MJ}x"
echo "* Pure Python wurde mit N=128 gemessen und auf N=1024 hochgerechnet (n^3)."

# ----------------------------------------------------------------------
# Quantized Matmul (4-bit Dequantization)
# ----------------------------------------------------------------------
NQ_MS=$(get RESULT_Q4_NUMPY_MS      "$PY_OUT")
NQ_GF=$(get RESULT_Q4_NUMPY_GFLOPS  "$PY_OUT")
N1_MS=$(get RESULT_Q4_NUMPY_1T_MS   "$PY_OUT")
N1_GF=$(get RESULT_Q4_NUMPY_1T_GFLOPS "$PY_OUT")
QU_MS=$(get RESULT_Q4_UNPACK_MS     "$PY_OUT")
QM_MS=$(get RESULT_Q4_MATMUL_MS     "$PY_OUT")
MQ_MS=$(get RESULT_Q4_MOJO_MS       "$MJ_OUT")
MQ_GF=$(get RESULT_Q4_MOJO_GFLOPS   "$MJ_OUT")
RB_MS=$(get RESULT_Q4_RB_MS         "$MJ_OUT")
RB_GF=$(get RESULT_Q4_RB_GFLOPS     "$MJ_OUT")
PK_MS=$(get RESULT_Q4_PK_MS         "$MJ_OUT")
PK_GF=$(get RESULT_Q4_PK_GFLOPS     "$MJ_OUT")
BP_MS=$(get RESULT_Q4_BP_MS         "$MJ_OUT")
BP_GF=$(get RESULT_Q4_BP_GFLOPS     "$MJ_OUT")
S1_MS=$(get RESULT_Q4_BP_1T_MS      "$MJ_OUT")
S1_GF=$(get RESULT_Q4_BP_1T_GFLOPS  "$MJ_OUT")
SM_MS=$(get RESULT_Q4_BP_MT_MS      "$MJ_OUT")
SM_GF=$(get RESULT_Q4_BP_MT_GFLOPS  "$MJ_OUT")
NCORES=$(get RESULT_CORES            "$MJ_OUT")
UNPACK_FRAC=$(calc "100.0 * $QU_MS / $NQ_MS")

# Speedups
SP_FUSED=$(calc "$NQ_MS / $MQ_MS")
SP_RB=$(calc    "$NQ_MS / $RB_MS")
SP_PK=$(calc    "$NQ_MS / $PK_MS")
SP_BP=$(calc    "$NQ_MS / $BP_MS")

# Scaling-Analyse
NP_SPEEDUP=$(calc  "$N1_MS / $NQ_MS")
NP_EFF=$(calc      "100.0 * $N1_MS / ($NQ_MS * $NCORES)")
MJ_SPEEDUP=$(calc  "$S1_MS / $SM_MS")
MJ_EFF=$(calc      "100.0 * $S1_MS / ($SM_MS * $NCORES)")

echo
echo "######################################################################"
echo "#       QUANTIZED MATMUL – OPTIMIERUNGSKETTE (4-bit, N=1024)        #"
echo "######################################################################"
printf "%-30s | %9s | %7s | %13s\n" "Backend" "Zeit[ms]" "GFLOPS" "vs NumPy MT"
printf -- "-------------------------------+-----------+---------+---------------\n"
printf "%-30s | %9s | %7s | %13s\n" "NumPy Q4 Expansion (MT)"      "$NQ_MS" "$NQ_GF"  "1.00 x"
printf "%-30s | %9s | %7s | %11s x\n" "Mojo Q4 Fused (naiv)"       "$MQ_MS" "$MQ_GF"  "$SP_FUSED"
printf "%-30s | %9s | %7s | %11s x\n" "Mojo Q4 RegBlocked 4x16"    "$RB_MS" "$RB_GF"  "$SP_RB"
printf "%-30s | %9s | %7s | %11s x\n" "Mojo Q4 A-Pack 6x16"        "$PK_MS" "$PK_GF"  "$SP_PK"
printf "%-30s | %9s | %7s | %11s x\n" "Mojo Q4 B-Pack (WINNER)"    "$BP_MS" "$BP_GF"  "$SP_BP"
echo
echo "Optimierungs-Kette (Mojo intern):"
echo "  Naiv -> RegBlocked: $(calc "$MQ_MS / $RB_MS")x   |   RegBlocked -> B-Pack: $(calc "$RB_MS / $BP_MS")x   |   Gesamt: $(calc "$MQ_MS / $BP_MS")x"
echo
echo "######################################################################"
echo "#     MULTITHREADING SCALING  (${NCORES} logische Kerne)                    #"
echo "######################################################################"
printf "%-28s | %9s | %8s | %8s | %11s\n" "Backend" "Zeit[ms]" "GFLOPS" "Threads" "Eff. (%)"
printf -- "-----------------------------+-----------+----------+---------+-------------\n"
printf "%-28s | %9s | %8s | %7s | %9s\n"  "NumPy Q4 1-Thread"       "$N1_MS" "$N1_GF"  "1"        "100.0"
printf "%-28s | %9s | %8s | %7s | %9s\n"  "NumPy Q4 ${NCORES}-Thread"         "$NQ_MS" "$NQ_GF"  "$NCORES"  "$NP_EFF"
printf "%-28s | %9s | %8s | %7s | %9s\n"  "Mojo Q4 B-Pack 1-Thread" "$S1_MS" "$S1_GF"  "1"        "100.0"
printf "%-28s | %9s | %8s | %7s | %9s\n"  "Mojo Q4 B-Pack ${NCORES}-Thread"   "$SM_MS" "$SM_GF"  "$NCORES"  "$MJ_EFF"
echo
echo "Speedup:  NumPy ${NP_SPEEDUP}x  |  Mojo ${MJ_SPEEDUP}x  (Ziel: ${NCORES}.00x bei perfekter Skalierung)"
echo "Mojo 1T GFLOPS ${S1_GF} -> ${NCORES}T GFLOPS ${SM_GF}"
echo "NumPy-Aufschluesselung: Unpack ${QU_MS} ms (${UNPACK_FRAC}%), BLAS-Matmul ${QM_MS} ms"
echo "B-Daten: NumPy 4 MiB (fp32) vs. Mojo 0.5 MiB (uint8) = 8x weniger Bandbreite."

package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMGAOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvSAT;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorObjectiveFunctionWrapper;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMessyBinaryVectorSpace;

/**
 * An example of using the Messy Genetic Algorithm Operator, it solves example
 * SAT formula.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */
public class EvMGAOperatorSATExample {
  public static void main(String[] args) {

    /*
     * The example formula is one from SAT 2002 competition
     * (http://www.satcompetition.org) unif-c500-v250-s673243691 This formula
     * has 250 variables and 500 clausules.
     */
    EvSAT sat_function =
        new EvSAT(
            "c genAlea with seed 673243691\n"
                + "p cnf 250 500\n"
                + "50 136 36 0\n-250 -113 17 0\n236 -241 -219 0\n-25 -205 168 0\n"
                + "-12 90 -32 0\n42 97 -36 0\n-221 214 5 0\n64 85 -14 0\n"
                + "-174 139 -64 0\n-102 -156 149 0\n-173 18 118 0\n-152 -88 173 0\n"
                + "95 83 40 0\n233 134 90 0\n-177 -21 -132 0\n-20 -110 72 0\n"
                + "90 66 89 0\n-19 -224 -113 0\n-196 -225 -217 0\n26 6 -227 0\n"
                + "216 -190 -32 0\n52 -235 48 0\n-106 6 166 0\n-92 -214 -191 0\n"
                + "-131 -74 -12 0\n23 46 52 0\n224 199 -221 0\n-32 62 155 0\n"
                + "-155 89 62 0\n-49 245 -79 0\n172 -233 65 0\n-44 139 -247 0\n"
                + "-239 -186 65 0\n28 -86 -87 0\n-145 90 8 0\n145 -21 146 0\n"
                + "-163 -123 183 0\n-160 147 221 0\n185 -180 -114 0\n168 -152 148 0\n"
                + "-175 87 171 0\n30 173 -134 0\n-20 -25 139 0\n127 -243 94 0\n"
                + "-72 -115 -18 0\n131 55 -138 0\n201 -58 -68 0\n-228 235 4 0\n"
                + "-248 198 -187 0\n-96 -52 231 0\n223 -11 80 0\n1 -65 153 0\n"
                + "250 -220 133 0\n108 104 -119 0\n-96 -48 -175 0\n-217 -140 -174 0\n"
                + "125 -250 232 0\n72 -33 -215 0\n14 44 -128 0\n157 -127 -121 0\n"
                + "233 -199 64 0\n92 -45 -128 0\n244 -8 240 0\n190 45 120 0\n"
                + "-146 145 230 0\n-73 -102 -85 0\n-143 -104 17 0\n214 180 -36 0\n"
                + "194 -172 -209 0\n160 67 125 0\n-69 -9 219 0\n-85 18 -48 0\n"
                + "169 -45 -234 0\n-18 -55 165 0\n230 116 128 0\n92 189 -117 0\n"
                + "193 -18 139 0\n245 -78 -156 0\n119 238 112 0\n186 227 136 0\n"
                + "42 -51 78 0\n-137 118 230 0\n-221 63 -13 0\n-109 -142 -54 0\n"
                + "-248 93 -40 0\n-78 207 -62 0\n45 -53 -103 0\n168 7 -120 0\n"
                + "199 -150 -75 0\n-93 193 -47 0\n-165 64 2 0\n-154 73 208 0\n"
                + "239 247 105 0\n14 -95 144 0\n-116 241 -172 0\n-40 -161 -206 0\n"
                + "228 -210 -66 0\n226 -42 49 0\n-89 200 -224 0\n102 -79 148 0\n"
                + "-130 -86 41 0\n-203 151 69 0\n-79 -20 16 0\n-17 -129 -38 0\n"
                + "-59 192 -232 0\n138 -27 151 0\n14 -86 -15 0\n129 199 214 0\n"
                + "215 69 -197 0\n-196 -245 143 0\n226 -88 -3 0\n69 -92 179 0\n"
                + "-58 -200 -206 0\n-221 -144 14 0\n-145 95 -96 0\n116 18 -73 0\n"
                + "-191 -36 -45 0\n138 22 -109 0\n-1 -147 118 0\n-234 -119 -151 0\n"
                + "-67 36 -218 0\n-104 250 190 0\n102 -148 166 0\n-100 -85 -61 0\n"
                + "156 214 124 0\n239 130 -23 0\n-133 -5 -15 0\n-177 23 240 0\n"
                + "-80 -104 85 0\n-117 -208 16 0\n195 21 -29 0\n171 170 -43 0\n"
                + "-120 50 -43 0\n-80 127 -38 0\n-97 -202 235 0\n136 221 3 0\n"
                + "200 -160 98 0\n-52 171 -110 0\n139 -86 -122 0\n-227 -27 -23 0\n"
                + "146 -191 65 0\n115 132 -226 0\n180 34 109 0\n228 188 -67 0\n"
                + "68 94 -105 0\n-249 106 -238 0\n-77 245 65 0\n141 116 33 0\n"
                + "-96 -169 110 0\n-198 -79 18 0\n-31 229 79 0\n106 68 113 0\n"
                + "-69 -125 -177 0\n-160 -168 56 0\n64 -204 45 0\n77 -238 210 0\n"
                + "20 -222 165 0\n-219 88 168 0\n-1 -37 -155 0\n196 96 -112 0\n"
                + "-121 -215 -242 0\n-174 -240 99 0\n230 137 -160 0\n-81 -132 79 0\n"
                + "208 -5 173 0\n-193 -54 -151 0\n82 -57 -220 0\n225 222 -16 0\n"
                + "166 -42 -212 0\n-83 -145 -223 0\n203 -111 -73 0\n-52 -87 -226 0\n"
                + "13 5 61 0\n116 115 22 0\n-29 246 191 0\n240 133 -150 0\n"
                + "-113 -110 -59 0\n5 -35 -55 0\n28 70 -78 0\n219 -196 -29 0\n"
                + "-182 30 120 0\n-21 131 180 0\n215 11 -193 0\n-94 -236 113 0\n"
                + "-5 -170 46 0\n87 -217 42 0\n-46 68 -192 0\n-116 -182 -156 0\n"
                + "-192 94 -163 0\n-241 -38 118 0\n108 60 71 0\n161 164 43 0\n"
                + "226 -12 -35 0\n30 167 -153 0\n80 -248 -53 0\n-200 -70 -236 0\n"
                + "-111 6 -124 0\n-129 207 6 0\n-57 -217 160 0\n-53 13 -107 0\n"
                + "-166 -226 66 0\n-125 -107 -200 0\n-178 -80 -218 0\n126 122 -248 0\n"
                + "-212 -20 -101 0\n39 -182 1 0\n-225 -173 5 0\n62 200 -67 0\n"
                + "227 -192 34 0\n-129 241 220 0\n-25 108 22 0\n-204 234 -78 0\n"
                + "248 -170 31 0\n-114 178 -106 0\n245 -206 -133 0\n76 183 114 0\n"
                + "41 202 193 0\n-11 -168 -137 0\n-167 206 55 0\n55 45 -34 0\n"
                + "-185 -85 41 0\n-72 -15 -104 0\n-52 -148 -46 0\n-169 -34 -56 0\n"
                + "184 138 -120 0\n156 -30 -172 0\n-188 151 17 0\n-161 -120 -39 0\n"
                + "-92 127 81 0\n-214 -43 -243 0\n-23 217 138 0\n-123 132 -29 0\n"
                + "29 -217 -246 0\n63 213 -212 0\n-122 -57 28 0\n-76 -197 -103 0\n"
                + "-180 -161 249 0\n-210 -128 175 0\n182 -211 178 0\n-186 -98 -214 0\n"
                + "-41 175 81 0\n-72 -4 71 0\n-171 30 -206 0\n-172 -192 -31 0\n"
                + "-55 169 209 0\n-27 171 236 0\n229 -11 40 0\n-235 111 9 0\n"
                + "-218 205 63 0\n29 -203 -19 0\n-79 160 -247 0\n250 203 -208 0\n"
                + "-91 202 -205 0\n-19 142 -82 0\n-209 72 -58 0\n-72 136 -135 0\n"
                + "-126 -107 -144 0\n70 5 -186 0\n233 133 178 0\n78 -75 48 0\n"
                + "240 18 -100 0\n-6 -36 53 0\n-176 163 144 0\n-141 54 -250 0\n"
                + "237 118 228 0\n193 -34 -91 0\n-113 -108 -209 0\n-77 -33 -66 0\n"
                + "-89 53 -229 0\n124 -131 26 0\n157 -75 -179 0\n-74 -12 -189 0\n"
                + "92 105 -4 0\n-51 126 -234 0\n156 69 99 0\n144 64 196 0\n"
                + "-134 33 200 0\n-115 -235 -164 0\n-140 81 -67 0\n119 172 -8 0\n"
                + "33 -25 121 0\n88 -238 -148 0\n90 -228 233 0\n-63 -105 -137 0\n"
                + "208 144 -194 0\n-149 54 72 0\n-142 -177 -28 0\n-67 -88 191 0\n"
                + "-185 191 147 0\n211 -83 232 0\n-5 1 -53 0\n-231 -74 232 0\n"
                + "125 -184 139 0\n-69 158 -156 0\n-69 222 -14 0\n-180 34 86 0\n"
                + "176 -172 74 0\n140 -223 -14 0\n191 48 216 0\n84 -166 -163 0\n"
                + "-111 77 211 0\n-13 -2 37 0\n35 214 -121 0\n-131 -3 -157 0\n"
                + "-45 -155 88 0\n-55 -13 145 0\n-135 106 -243 0\n-28 152 76 0\n"
                + "-146 -82 -135 0\n116 -229 139 0\n86 -15 -76 0\n-72 -67 96 0\n"
                + "-69 -86 114 0\n-242 117 157 0\n62 -180 -42 0\n107 12 -226 0\n"
                + "-30 47 -224 0\n155 -15 -12 0\n152 -138 76 0\n20 66 -178 0\n"
                + "19 -234 154 0\n-118 -5 29 0\n56 20 173 0\n-199 34 -93 0\n"
                + "130 -246 62 0\n52 -206 110 0\n-243 43 -141 0\n-225 -3 18 0\n"
                + "25 -250 53 0\n-167 -205 127 0\n85 -97 -212 0\n151 184 201 0\n"
                + "-13 21 -169 0\n-192 -5 1 0\n-34 -238 79 0\n-5 42 -205 0\n"
                + "-58 -208 48 0\n65 84 -76 0\n125 -196 208 0\n-185 -113 229 0\n"
                + "37 105 85 0\n-82 -165 132 0\n-198 22 33 0\n-33 -113 -36 0\n"
                + "174 -62 -143 0\n-227 24 8 0\n140 249 -129 0\n15 78 106 0\n"
                + "-133 17 116 0\n66 22 75 0\n-168 -94 131 0\n48 -191 75 0\n"
                + "-178 -145 -126 0\n115 139 -4 0\n184 44 -197 0\n120 168 55 0\n"
                + "-64 -70 149 0\n73 -233 -80 0\n217 -116 -77 0\n111 189 -216 0\n"
                + "22 249 -48 0\n-239 155 114 0\n201 129 153 0\n41 153 -226 0\n"
                + "25 -240 98 0\n40 202 150 0\n-198 248 -108 0\n-248 131 194 0\n"
                + "21 -78 -77 0\n-82 -66 -21 0\n83 11 148 0\n224 -214 157 0\n"
                + "-169 -57 -55 0\n-221 -97 -71 0\n151 -239 224 0\n-130 70 91 0\n"
                + "-49 62 -35 0\n83 -122 -3 0\n139 1 22 0\n159 -223 138 0\n"
                + "157 2 156 0\n192 -156 217 0\n-221 145 188 0\n-19 -120 62 0\n"
                + "91 63 117 0\n10 216 -249 0\n-104 -76 193 0\n201 91 6 0\n"
                + "-242 104 43 0\n2 -135 -197 0\n-14 -24 217 0\n-100 212 80 0\n"
                + "-217 68 -124 0\n-85 -96 -226 0\n-103 -22 42 0\n-238 -173 -34 0\n"
                + "33 210 58 0\n-150 -69 81 0\n-40 58 103 0\n-24 -12 191 0\n"
                + "13 108 190 0\n-75 24 -217 0\n74 -106 -83 0\n86 102 7 0\n"
                + "178 -159 -76 0\n242 -28 23 0\n240 189 -131 0\n95 -22 -98 0\n"
                + "136 226 77 0\n191 -13 -193 0\n-199 -175 -33 0\n-193 -44 -37 0\n"
                + "38 129 81 0\n-163 -159 204 0\n-120 -239 -244 0\n220 -115 169 0\n"
                + "-92 170 -243 0\n-59 87 -88 0\n-164 135 138 0\n-59 233 140 0\n"
                + "228 169 5 0\n239 -199 230 0\n115 221 -227 0\n-120 -59 -213 0\n"
                + "93 -204 -68 0\n52 -203 -137 0\n58 -52 34 0\n-196 70 101 0\n"
                + "225 -186 -224 0\n229 -170 -142 0\n25 -32 -164 0\n183 43 226 0\n"
                + "16 225 -88 0\n-132 46 94 0\n217 86 -4 0\n-122 -170 4 0\n"
                + "71 -53 171 0\n190 -200 -23 0\n-240 99 222 0\n54 -15 -73 0\n"
                + "-116 -46 98 0\n-118 237 -32 0\n199 94 136 0\n185 20 161 0\n"
                + "226 178 104 0\n9 -196 5 0\n-104 -212 -42 0\n-247 215 -138 0\n"
                + "206 -226 -106 0\n-215 -101 -72 0\n143 174 134 0\n132 -50 -77 0\n"
                + "86 -83 163 0\n150 -43 -92 0\n12 230 199 0\n67 -130 180 0\n"
                + "168 -250 154 0\n-60 149 -40 0\n-112 -178 -65 0\n-53 -77 -189 0\n"
                + "82 -117 -245 0\n-225 108 130 0\n50 -1 -81 0\n9 -229 -97 0\n"
                + "207 119 -180 0\n57 62 -224 0\n-221 218 -172 0\n-89 -37 -170 0\n"
                + "129 -60 115 0\n-149 54 -104 0\n75 246 -84 0\n-192 -166 -132 0\n"
                + "-7 59 -60 0\n160 157 44 0\n-124 -93 210 0\n162 106 171 0\n"
                + "-125 -244 22 0\n128 -95 -71 0\n165 92 -198 0\n93 77 249 0\n"
                + "131 72 -67 0\n176 -104 231 0\n43 153 45 0\n42 -203 -31 0\n"
                + "18 222 -58 0\n-132 -131 175 0\n195 -138 -18 0\n-174 228 -199 0\n"
                + "-49 -231 -124 0\n152 134 236 0\n-95 130 69 0\n-173 154 61 0\n"
                + "-180 232 60 0\n178 170 110 0\n170 -184 -135 0\n-245 -36 -205 0\n"
                + "178 118 188 0\n-2 163 23 0\n-163 81 110 0\n-126 201 82 0\n");

    // Parameters of the example

    /* Number of eras */
    final int maximum_era = 3;
    /* Size of the problem, length of vector */
    final int problem_size = sat_function.getVariablesNumber();
    /*
     * Upper limit of the population size, 0 means that there is no limit. If
     * generated cover individuals number exceedes this, an uniformly random
     * individuals will be selected from them, this option can preserve from out
     * of memory due to extreme big populations, so it enables using more eras,
     * NOTE: this option does not belong to the original mGA
     */
    final int maximum_population_size = 100000;
    /*
     * Probability of cut an individual, multiplied by the length of the
     * chromosome, recommended 1.0 / (2 * problem_size) value
     */
    final double probability_of_cut = 2.0 / (2 * problem_size);
    /* Probability of splice two individuals, recommended high values or 1.0 */
    final double probability_of_splice = 0.8;
    /*
     * Probability of allele negation for each allele, recommended small or 0.0
     */
    final double probability_of_allelic_mutation = 0.02;
    /*
     * Probability of change gene which allele belongs, recommended small or
     * 0.0, NOTE: this not guarantying changing gene to a different one, for
     * probability guarantying changing gene use genic_mutation =
     * changing_genic_mutation * (problem_length / (problem_length-1)), in the
     * original mGA guarantying changing gene mutation is used
     */
    final double probability_of_genic_mutation = 0.0;
    /*
     * There will be compared individuals with a number of common expressed
     * genes larger than expected in random chromosomes
     */
    final boolean thresholding = false;
    /*
     * Shorter individuals have advantage when the objective function value is
     * the same
     */
    final boolean tie_breaking = true;
    /*
     * Negated template is used for generated individuals instead all allele
     * combinations.
     */
    final boolean reduced_initial_population = true;
    /*
     * Find and keep for the best individual in whole era time, instead of get
     * it from final era population, NOTE: this option is an experimental
     * extension, it does not belong to the original mGA.
     */
    final boolean keep_era_best_individual = false;
    /*
     * This array contains the number of duplicates of each individual in the
     * initial population for each era
     */
    final int[] copies = new int[] {5, 1, 1};
    /* Number of generations, specified for all eras */
    final int[] maximum_generationes = new int[] {10, 10, 10};
    /* Population size in the juxtapositional phase, specified for each era */
    final int[] juxtapositional_sizes = new int[] {100, 50, 50};

    EvMGAOperator mga_operator =
        new EvMGAOperator(maximum_era, problem_size, maximum_population_size,
            probability_of_cut, probability_of_splice,
            probability_of_allelic_mutation, probability_of_genic_mutation,
            thresholding, tie_breaking, reduced_initial_population,
            keep_era_best_individual, copies, maximum_generationes,
            juxtapositional_sizes);

    // Create the algorithm
    EvAlgorithm<EvMessyBinaryVectorIndividual> messyGA =
        new EvAlgorithm<EvMessyBinaryVectorIndividual>(1);

    EvMessyBinaryVectorObjectiveFunctionWrapper objective_function =
        new EvMessyBinaryVectorObjectiveFunctionWrapper(sat_function);

    messyGA.setSolutionSpace(new EvMessyBinaryVectorSpace(objective_function,
        problem_size));

    int iteration_number = 0;
    for (int i = 0; i < maximum_era; i++)
      iteration_number += maximum_generationes[i];
    messyGA
        .setTerminationCondition(new EvMaxIteration<EvMessyBinaryVectorIndividual>(
            iteration_number));

    messyGA.addOperatorToEnd(mga_operator);

    messyGA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvMessyBinaryVectorIndividual>(
            System.out));

    // Run the algorithm
    EvTask task = new EvTask();
    task.setAlgorithm(messyGA);
    long startTime = System.currentTimeMillis();
    task.run();
    long endTime = System.currentTimeMillis();
    System.out.println("Total time: " + ((double) endTime - startTime) / 1000
        + "s");
  }

}
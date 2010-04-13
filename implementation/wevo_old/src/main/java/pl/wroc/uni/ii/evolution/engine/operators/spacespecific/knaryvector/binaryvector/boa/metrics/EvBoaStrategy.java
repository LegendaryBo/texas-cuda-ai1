package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.metrics;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork.EvBayesianNetwork;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvBinary;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.utils.EvTriple;

public class EvBoaStrategy {

  private double metric[];

  private double gain[][];

  private EvBinary ev_binary;

  private int k;

  private int[] m_parents;

  private int[] m_true;

  private int[] m_false;


  public EvBoaStrategy() {
  }


  public void init(int n, int k) {
    // inial binary tools
    ev_binary = new EvBinary();
    ev_binary.precompute(n + 1);
    ev_binary.precomputeBoolsTable(k);
    this.k = k;
  }


  public void model(EvPopulation<EvBinaryVectorIndividual> population,
      EvBayesianNetwork network) {

    int n = network.getN();
    // alloc space
    metric = new double[n];
    gain = new double[n][n];

    // convert to table
    EvBinaryVectorIndividual[] array =
        population.toArray(new EvBinaryVectorIndividual[population.size()]);

    // inital metric
    for (int i = 0; i < n; i++) {
      metric[i] = computeMetric3(i, network.getParentsIndexes(i), array);
    }

    // compute inital gains
    boolean[] legal = new boolean[n];
    for (int i = 0; i < n; i++) {

      for (int j = 0; j < n; j++) {
        if (network.legalEdge(i, j)) {
          legal[j] = true;
        } else {
          legal[j] = false;
        }
      }

      int[] parent_indexes = network.getParentsIndexes(i);
      double[] metrics = computeMetics(i, parent_indexes, legal, array);

      for (int j = 0; j < n; j++) {
        if (legal[j]) {
          gain[i][j] = metrics[j] - metric[i];
        } else {
          gain[i][j] = Integer.MIN_VALUE;
        }
      }
    }

    // adding some edges (maximum == network.k * network.n)
    int count = 0;

    while (count < (k * n)) {

      // search for best edge addition which improves net
      int best_i = 0, best_j = 0;
      double best = Double.NEGATIVE_INFINITY;

      // find best edge addition
      for (int i = 0; i < n; i++) {

        if (network.getParentsIndexesCount(i) < k) {

          for (int j = 0; j < n; j++) {
            if ((gain[i][j] > 0) && (gain[i][j] > best)) {
              best_i = i;
              best_j = j;
              best = gain[i][j];
            }
          }
        }
      }

      if (best > 0) {

        // add best edge to net
        network.addEdge(best_i, best_j);

        // recompute metric of node
        metric[best_i] =
            computeMetric3(best_i, network.getParentsIndexes(best_i), array);

        // /// recompute all gains for node best_i

        // check if node is legal
        for (int j = 0; j < n; j++) {
          if (network.legalEdge(best_i, j)) {
            legal[j] = true;
          } else {
            legal[j] = false;
          }
        }

        //
        int[] parent_indexes = network.getParentsIndexes(best_i);
        double[] metrics = computeMetics(best_i, parent_indexes, legal, array);
        for (int j = 0; j < n; j++) {
          if (legal[j]) {
            gain[best_i][j] = metrics[j] - metric[best_i];
          } else {
            gain[best_i][j] = Integer.MIN_VALUE;
          }
        }
      } else {
        break;
      }
    }

  }


  public double computeMetric3(int i, int[] parents_indexes,
      EvBinaryVectorIndividual[] population) {

    double result = 0.0;
    EvTriple[] trips = EvBinary.numberOf(population, parents_indexes, i);

    for (int k = 0; k < trips.length; k++) {
      result += ev_binary.sumLogFromMap(1 + trips[k].x);
      result += ev_binary.sumLogFromMap(1 + trips[k].y);
      result -= ev_binary.sumLogFromMap(1 + trips[k].z);
    }
    return result;

  }


  public double[] computeMetics(int i, int[] start_parents, boolean[] legal,
      EvBinaryVectorIndividual[] array) {

    double[] result = new double[legal.length];
    int len = start_parents.length;
    int len_pow = EvBinary.pow2(len);

    this.m_parents = new int[2 * len_pow * legal.length];
    this.m_true = new int[2 * len_pow * legal.length];
    this.m_false = new int[2 * len_pow * legal.length];

    // iterate over indiviuals
    int array_size = array.length;
    EvBinaryVectorIndividual local_individual;
    int k;

    for (k = 0; k < array_size; k++) {
      local_individual = array[k];

      int s = 0;
      int x = 1;

      for (int l = 0; l < len; l++) {
        // System.out.print(local_vector[start_parents[l]] ? "1" : "0");
        if (local_individual.getGene(start_parents[l]) == 1) {
          s += x;
        }
        x *= 2;
      }
      // System.out.println("\n" + s);
      // System.out.println("\n---\n");

      for (int r = 0; r < legal.length; r++) {
        if (legal[r]) {

          if (local_individual.getGene(r) == 0) {

            m_parents[r * 2 * len_pow + s]++;
            if (local_individual.getGene(i) == 1) {
              m_true[r * 2 * len_pow + s]++;
            } else {
              m_false[r * 2 * len_pow + s]++;
            }
          } else {

            m_parents[r * 2 * len_pow + s + len_pow]++;
            if (local_individual.getGene(i) == 1) {
              m_true[r * 2 * len_pow + s + len_pow]++;
            } else {
              m_false[r * 2 * len_pow + s + len_pow]++;
            }
          }
        }
      }
    }

    int tmp;
    for (int m = 0; m < legal.length; m++) {
      result[m] = 0;
      for (int p = 0; p < 2 * len_pow; p++) {
        tmp = m * 2 * len_pow + p;

        result[m] += ev_binary.sumLogFromMap(1 + m_true[tmp]);
        result[m] += ev_binary.sumLogFromMap(1 + m_false[tmp]);
        result[m] -= ev_binary.sumLogFromMap(1 + m_parents[tmp]);
      }
    }

    return result;
  }
}
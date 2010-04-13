package pl.wroc.uni.ii.evolution.experimental;



import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork.EvBinary;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork.EvBayesianNetwork;

/**
 * @author Jarek Fuks
 * @author Zbigniew Nazimek 
 * 
 * Bayesian-Dirchlet metric without prior knowlege about problem
 */
public class EvBayesianDirchletMetric {

  /**
   * Rates given network.
   * 
   * @param population - population which takes part in computation.
   * @param network - bayesian network to be computed
   * @param ev_binary - ?
   * @return rate of given network.
   */
  public double compute(
      final EvPopulation<EvBinaryVectorIndividual> population,
      final EvBayesianNetwork network, 
      final EvBinary ev_binary) {
    double value = 0.0;
    int number_of_nodes = population.get(0).getDimension();

    for (int i = 0; i < number_of_nodes; i++) {
      int[] parents_indexes = network.getParentsIndexes(i);
      int[] all_indexes = new int[parents_indexes.length + 1];

      for (int n = 0; n < parents_indexes.length; n++) {
        all_indexes[n] = parents_indexes[n];
      }
      all_indexes[all_indexes.length - 1] = i;

      // this can be optimalised
      double sub_value = 0.0;
      for (int k = 0; k < EvBinary.pow2(parents_indexes.length); k++) {
        int[] parents_value = EvBinary.intToBools(k, parents_indexes.length);
        int[] all_value = EvBinary.intToBools(k, parents_indexes.length + 1);

        all_value[all_value.length - 1] = 0;
        int m_parents =
            EvBinary.numberOf(population, parents_value, parents_indexes);
        int m_false = EvBinary.numberOf(population, all_value, all_indexes);
        all_value[all_value.length - 1] = 1;
        int m_true = EvBinary.numberOf(population, all_value, all_indexes);

        sub_value += ev_binary.sumLogFromMap(1 + m_true);
        sub_value += ev_binary.sumLogFromMap(1 + m_false);
        sub_value -= ev_binary.sumLogFromMap(1 + m_parents);

      }
      value += sub_value;
    }
    return value;
  }

}

package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork;

/**
 * @author Jarek Fuks, Zbigniew Nazimek
 */

import java.util.HashMap;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class EvBinary {

  private HashMap<Integer, Double> map = new HashMap<Integer, Double>();

  private int biggest_log_in_map = 0;


  public EvBinary() {
    map.put(0, 0.0);
  }


  /**
   * @param population
   * @param values
   * @param indexes
   * @return
   */
  public static int numberOf(EvPopulation<EvBinaryVectorIndividual> population,
      int[] values, int[] indexes) {
    int ret = 0;
    /*
     * @@@@@@NOTE@@@@@@ DONT PUT ANY SUGAR HERE BOA spends most his time in this
     * function and it must use as less CPU as possible!!!
     */
    if (values.length == 0)
      return population.size();

    // creating individual table for better performance
    EvBinaryVectorIndividual[] ind_table =
        population.toArray(new EvBinaryVectorIndividual[population.size()]);
    int pop_size = population.size();

    boolean compare = true; // for comparing
    int j = 0; // for comparing
    int indexes_length = indexes.length; // for comparing

    int k = 0;
    int i = 0;

    for (i = 0; i < pop_size; i++) {

      // public static boolean compare - for better perfomance
      j = 0;
      compare = true;
      for (k = 0; k < indexes_length; k++) {
        if (ind_table[i].getGene(indexes[k]) != values[j++]) {
          compare = false;
          break;
        }
      }
      // end of public static boolean compare

      if (compare)
        ret++;
    }

    return ret;
  }


  // deprecated - it is implemented in numberOf
  public static boolean compare(EvBinaryVectorIndividual x, int[] a,
      int[] indeses) {
    int j = 0;
    int indeses_length = indeses.length;
    for (int i = 0; i < indeses_length; i++) {
      if (x.getGene(indeses[i]) != a[j++])
        return false;
    }
    return true;
  }


  public static int[] intToBools(int bools, int size) {
    int[] ret = new int[size];

    for (int i = 0; i < size; i++) {
      if (bools % 2 != 0) {
        ret[i] = 1;
      } else {
        ret[i] = 0;
      }
      bools /= 2;
    }
    return ret;
  }


  public static int pow2(int n) {
    return new int[] {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
        4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824}[n];
  }


  public static double factorial(int n) {
    double ret = 1.0;
    for (int i = 2; i <= n; i++) {
      ret *= i;
    }
    return ret;
  }


  public double sumLogFromMap(int n) {
    Double logn = map.get(n);
    if (logn == null) {
      double new_value = 0.0;
      for (int i = biggest_log_in_map + 1; i <= n; i++) {
        new_value = Math.log(i) + map.get(i - 1);
        map.put(i, new_value);

      }
      biggest_log_in_map = n;
      return new_value;
    } else
      return logn;
  }


  public static double sumLog(int n) {
    double ret = 0.0;
    for (int i = 1; i <= n; i++)
      ret += Math.log(i);
    return ret;
  }

}

package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Implementation for a roulette selection with custom fitness function. In this
 * selection operator an individual has a chance to be selected propotional to
 * its fitness value.
 * 
 * @author Marcin Golebiowski, Kamil Dworakowski
 */

public class EvRouletteSelection<T extends EvIndividual> extends EvSelection<T> {

  private EvIndividualFitness<T> fit;

  private int k;


  /**
   * @param fitness function to rate individuals
   * @param k number of invidiual to select
   */
  public EvRouletteSelection(EvIndividualFitness<T> fitness, int k) {
    this.fit = fitness;
    this.k = k;

  }

  static class IndivInfo<T> {
    T indiv;

    double fitness;


    IndivInfo(T indiv, double fitness) {
      this.indiv = indiv;
      this.fitness = fitness;
    }
  }

  public static class RouletteWheel<T extends EvIndividual> {
    List<IndivInfo<T>> infos;

    double all_val_fitness = 0;


    public RouletteWheel(EvPopulation<T> population, EvIndividualFitness<T> fit) {

      infos = new ArrayList<IndivInfo<T>>(population.size());

      for (T indiv : population) {
        infos.add(new IndivInfo<T>(indiv, fit.getFitness(indiv)));
      }

      Collections.sort(infos, new Comparator<IndivInfo>() {
        public int compare(IndivInfo arg0, IndivInfo arg1) {
          return (int) Math.signum(arg1.fitness - arg0.fitness);
        }
      });

      for (IndivInfo info : infos) {
        all_val_fitness += info.fitness;
      }
    }


    /**
     * Determines, which individual (pie part) occupies the spece that contains
     * the position.
     * 
     * @param position should be in range [0,1]
     * @return
     */
    public int getIndivIndexAtPosition(double position) {
      double target = position * all_val_fitness;
      double sum_so_far = 0.0;

      int infos_size = infos.size();

      for (int i = 0; i < infos_size; i++) {
        sum_so_far += infos.get(i).fitness;
        if (target < sum_so_far) {
          return i;
        }
      }
      return infos.size() - 1;
    }


    /**
     * Draws an indiviudal from the roulette wheel.
     * 
     * @return
     */
    public int drawIndexOfIndividual() {
      return getIndivIndexAtPosition(EvRandomizer.INSTANCE.nextDouble());
    }
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {
    List<Integer> result = new ArrayList<Integer>();
    fit.reinitialize(population);
    RouletteWheel<T> wheel = new RouletteWheel<T>(population, fit);
    for (int i = 0; i < k; i++) {
      result.add(wheel.drawIndexOfIndividual());
    }
    return result;
  }
}
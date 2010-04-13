package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * How this selection works: <br> - first selection choose some individuals from
 * population <br> - second selection filter those individuals selected by the
 * first operator.<br>
 * As a result output population is filtered by two selction operator.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvTwoSelectionComposition<T extends EvIndividual> extends
    EvSelection<T> {

  private EvSelection<T> first;

  private EvSelection<T> second;


  /**
   * Constructor
   * 
   * @param first_selection
   * @param second_selection
   */
  public EvTwoSelectionComposition(EvSelection<T> first_selection,
      EvSelection<T> second_selection) {
    this.first = first_selection;
    this.second = second_selection;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {
    List<Integer> result = new ArrayList<Integer>();

    List<Integer> tmp = first.getIndexes(population);

    EvPopulation<T> first_result = new EvPopulation<T>();
    for (Integer i : tmp) {
      first_result.add(population.get(i));
    }

    List<Integer> tmp2 = second.getIndexes(first_result);

    for (Integer selected : tmp2) {
      result.add(tmp.get(selected));

    }

    return result;
  }

}

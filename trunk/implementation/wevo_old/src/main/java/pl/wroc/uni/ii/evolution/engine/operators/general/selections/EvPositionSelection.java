package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * Selects individuals with indexes 0 ... how_many
 * 
 * @author Marcin Golebiowski
 */
public class EvPositionSelection<T extends EvIndividual> extends EvSelection<T> {

  private int how_many;


  /**
   * @param how_many how many following individuals are selected
   */
  public EvPositionSelection(int how_many) {
    this.how_many = how_many;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {

    if (how_many > population.size()) {
      throw new IllegalStateException(
          "EvPositionSelection: population too small");
    }

    List<Integer> result = new ArrayList<Integer>();

    for (int i = 0; i < how_many; i++) {
      result.add(i);
    }

    return result;
  }

}

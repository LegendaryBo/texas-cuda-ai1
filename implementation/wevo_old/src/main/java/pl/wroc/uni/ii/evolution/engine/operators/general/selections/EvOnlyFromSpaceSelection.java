package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * Returns filtered population that contains only individuals from given
 * EvSolutionSpace
 * 
 * @author Marcin Golebiowski
 */

public class EvOnlyFromSpaceSelection<T extends EvIndividual> extends
    EvSelection<T> {

  private EvSolutionSpace<T> space;


  /**
   * Contructor
   * 
   * @param space solution space for checking individuals
   */
  public EvOnlyFromSpaceSelection(EvSolutionSpace<T> space) {
    this.space = space;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {
    List<Integer> result = new ArrayList<Integer>();

    for (int i = 0; i < population.size(); i++) {
      if (space.belongsTo(population.get(i))) {
        result.add(i);
      }
    }
    return result;
  }
}

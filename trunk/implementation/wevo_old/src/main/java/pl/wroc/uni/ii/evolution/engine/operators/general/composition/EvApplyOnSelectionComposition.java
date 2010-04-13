package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * EvApplyOnSelectionComposition first constructs two populations:<br> - first
 * by applying specified selection population to input population.<br> -
 * second, which contains the remaining individuals not selected by the
 * selection operator. Then another specified operator is applied on first
 * population and complement operator is applied on second population. At the
 * end both population are merged.<br>
 * <br>
 * If you don't want to process second population, use secondary constructor
 * (without the complement operator)
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvApplyOnSelectionComposition<T extends EvIndividual> implements
    EvOperator<T> {

  private EvSelection<T> selection;

  private EvOperator<T> operator;

  private EvOperator<T> complement_operator;


  /**
   * Primary constructor.
   * 
   * @param selection operator which selects some individuals from input
   *        population
   * @param operator will be applied on selected individuals
   * @param complement_operator will we applied on unselected individuals
   */
  public EvApplyOnSelectionComposition(EvSelection<T> selection,
      EvOperator<T> operator, EvOperator<T> complement_operator) {
    this.selection = selection;
    this.operator = operator;
    this.complement_operator = complement_operator;
  }


  /**
   * Secondary constructor with no second operator. The second population won't
   * be processed as there is no operator and is not going to be merged with the
   * first population
   * 
   * @param selection selects individuals
   * @param operator will be applied on selected individuals
   */
  public EvApplyOnSelectionComposition(EvSelection<T> selection,
      EvOperator<T> operator) {
    this.selection = selection;
    this.operator = operator;
    this.complement_operator = null;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {

    List<Integer> selected_indexes = selection.getIndexes(population);
    EvPopulation<T> selected_individuals =
        EvSelection.apply(population, selected_indexes);
    EvPopulation<T> first_result = operator.apply(selected_individuals);

    // when there is no second operator
    if (complement_operator != null) {

      EvPopulation<T> second_result =
          complement_operator.apply(EvSelection.apply(population, EvSelection
              .getUnselected(selected_indexes, population.size())));

      // merging both populations
      for (T ind : second_result) {
        first_result.add(ind);
      }
    }

    return first_result;
  }

}

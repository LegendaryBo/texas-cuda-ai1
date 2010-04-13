package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * It creates children population using <b>children_operator</b> and using
 * <b>replacement</b> operator combine the new population with the old one.<br>
 * Example:<br>
 * consider onemax problem and input population:<br>
 * (1,1,1,1,1) ; (0,0,0,0,0)<br>
 * <br>
 * We apply EvReplacementComposition with EvEliteReplacement(2,1) and
 * EvBinaryVectorNegationMutation<br>
 * <br>
 * EvBinaryVectorNegationMutation created 2 individuals:<br>
 * (1,1,0,1,0) ; (0,0,1,1,0)<br>
 * <br>
 * Output population:<br>
 * (1,1,1,1,1) ; (1,1,0,1,0)<br>
 * 
 * @author Zbigniew Nazimek (???)
 * @author Jarek Fuks (jarek102@gmail.com)
 */
public class EvReplacementComposition<T extends EvIndividual> implements
    EvOperator<T> {

  private EvOperator<T> children_operator;

  private EvReplacement<T> replacement;


  /**
   * Constructor.
   * 
   * @param children_operator creates children of population
   * @param replacement using input population and children creates result
   *        population
   */
  public EvReplacementComposition(EvOperator<T> childrent_operator,
      EvReplacement<T> replacement) {
    this.children_operator = childrent_operator;
    this.replacement = replacement;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {
    return replacement.apply(new EvPopulation<T>(population), children_operator
        .apply(population));
  }

}

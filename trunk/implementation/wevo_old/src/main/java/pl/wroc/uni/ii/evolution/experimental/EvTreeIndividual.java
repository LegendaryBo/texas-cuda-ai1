package pl.wroc.uni.ii.evolution.experimental;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.utils.EvIRandomizer;

/**
 * A super class individual that allows to make general tree operators. Tree
 * individuals could inherit from this class to utilize present tree operators.
 * 
 * @author Kamil Dworakowski
 * @param <T> a type of specific TreeIndividual
 */
public abstract class EvTreeIndividual<T extends EvTreeIndividual> extends
    EvIndividual {

  /**
   * Replaces subtree. Matching what_to_replace with a proper node to replace
   * should be made pointerwise.
   * 
   * @return clone of this individual with what_to_replace replaced by
   *         substitute
   */
  public abstract T replace(T what_to_replace, T substitute);


  /**
   * Should return random descendant. It could be the root itself.
   * 
   * @param randomizer an object that will be used to draw a random number
   * @return randomly chosen descendant
   */
  public abstract T randomDescendant(EvIRandomizer randomizer);
}

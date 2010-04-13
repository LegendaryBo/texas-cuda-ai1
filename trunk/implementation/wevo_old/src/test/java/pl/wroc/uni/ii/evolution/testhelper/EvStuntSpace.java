
package pl.wroc.uni.ii.evolution.testhelper;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

public class EvStuntSpace implements EvSolutionSpace<EvStunt> {

  private static final long serialVersionUID = -1862227270066022254L;

  private int cur;

  private int parity;

  /**
   * 
   * @param parity
   */
  public EvStuntSpace( int parity ) {
    this.cur = 0;
    this.parity = parity;
  }
  /**
   * Check if individual given with argument <code>individual</code> 
   * belongs to this solution space. True if individual is binary and has the same dimension.
   * 
   * @return if individual belongs to this solution space 
   * 
   */
  public boolean belongsTo( EvStunt individual ) {
    return individual.gene % 2 == parity;
  }
  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvStunt>> divide( int n ) {
    return null;
  }
  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvStunt>> divide( int n, Set<EvStunt> p ) {
    return null;
  }
  
  /**
   * @return 
   */
  public EvStunt generateIndividual() {
    EvStunt gen = new EvStunt( 2 * cur + parity );
    cur++;
    gen.setObjectiveFunction(new EvGoalFunction());
    return gen;
  }

  public EvStunt takeBackTo( EvStunt individual ) {
    return null;
  }
  public EvObjectiveFunction<EvStunt> getObjectiveFuntion() {
 
    return null;
  }
  public void setObjectiveFuntion(EvObjectiveFunction<EvStunt> objective_function) {
   
    
  }

}

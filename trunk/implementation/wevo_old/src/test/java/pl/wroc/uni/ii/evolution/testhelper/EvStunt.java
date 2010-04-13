package pl.wroc.uni.ii.evolution.testhelper;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * A test double of Individual. To ease testing.
 * 
 * @author Kamil Dworakowski
 *
 */
public class EvStunt extends EvIndividual {
  
  private static final long serialVersionUID = 1L;
  
  double gene;

  public EvStunt(double gene) {
    this.gene = gene;
  }

  @Override
  public Object clone() {
    EvStunt clone = new EvStunt(gene);
    clone.setObjectiveFunction(getObjectiveFunction());
    return clone;
  }

  @Override
  public int hashCode() {
    final int PRIME = 31;
    int result = 1;
    result = PRIME * result + (int)Double.doubleToLongBits(gene);
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final EvStunt other = (EvStunt) obj;
    if (gene!= other.gene)
      return false;
    return true;
  }

  @Override
  public String toString() {
    return Double.toString(gene);
  }

  public double getValue() {
    return gene;
  }
  
  public static List<EvStunt> list(double ... vals) {
    EvGoalFunction goalFunction = new EvGoalFunction();
    List<EvStunt> stunts = new ArrayList<EvStunt>(vals.length);
    for(double val : vals) {
      EvStunt stunt = new EvStunt(val);
      stunt.setObjectiveFunction(goalFunction);
      stunts.add(stunt);
    }
    return stunts;
  }
  
  public static Set<EvStunt> set(double ... vals) {
    return new TreeSet<EvStunt>(list(vals));
  }
  
  public static EvPopulation<EvStunt> pop(double ... vals) {
    return new EvPopulation<EvStunt>(list(vals));
  }
}

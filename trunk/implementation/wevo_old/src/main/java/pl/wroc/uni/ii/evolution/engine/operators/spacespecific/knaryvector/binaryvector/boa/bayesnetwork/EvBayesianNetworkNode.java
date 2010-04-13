package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.bayesnetwork;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Node in bayesian network
 * 
 * @author Marcin Golebiowski, Jarek Fuks , Zbigniew Nazimek
 */
public class EvBayesianNetworkNode {
  private int index;

  private List<EvBayesianNetworkNode> parents;

  private List<EvBayesianNetworkNode> children;

  private double[] probabilities;


  public EvBayesianNetworkNode(int index, int n) {
    this.index = index;
    this.parents = new ArrayList<EvBayesianNetworkNode>();
    this.children = new ArrayList<EvBayesianNetworkNode>();
    this.probabilities = new double[1];
    this.probabilities[0] = 0.5;
  }


  public int getIndex() {
    return this.index;
  }


  /* ========================================================== */

  public void addParent(EvBayesianNetworkNode parent) {
    this.parents.add(parent);
  }


  public void addChild(EvBayesianNetworkNode child) {
    this.children.add(child);
  }


  public void removeChild(EvBayesianNetworkNode child) {
    this.children.remove(child);
  }


  public void removeParent(EvBayesianNetworkNode parent) {
    this.parents.remove(parent);
  }


  public EvBayesianNetworkNode[] getParents() {
    EvBayesianNetworkNode[] ret = new EvBayesianNetworkNode[parents.size()];
    return ret = parents.toArray(ret);
  }


  public EvBayesianNetworkNode[] getChildren() {
    EvBayesianNetworkNode[] ret = new EvBayesianNetworkNode[children.size()];
    return ret = children.toArray(ret);
  }


  public boolean hasChild(EvBayesianNetworkNode node) {
    return this.children.contains(node);
  }


  public boolean hasParent(EvBayesianNetworkNode node) {
    return this.parents.contains(node);
  }


  public int getParentCount() {
    return parents.size();
  }


  public int getChildrenCount() {
    return children.size();
  }


  /* =============================================================================== */

  @Override
  public boolean equals(Object arg0) {
    if ((arg0 != null) && arg0.getClass().equals(EvBayesianNetworkNode.class)) {
      // System.out.println((EvBayesianNetworkNode) arg0);
      return ((EvBayesianNetworkNode) arg0).index == this.index;
    }

    return super.equals(arg0);
  }


  /* =============================================================================== */

  /**
   * set table of probabilities, which is using to generate values
   * 
   * @param probabilities
   */
  public void setProbabilities(double[] probabilities) {
    if (Math.pow(2, parents.size()) != probabilities.length)
      throw new IllegalArgumentException(
          "number of parents and table of probabilities not marged");
    this.probabilities = probabilities;
  }


  /**
   * generate new value (when valid is false)
   * 
   * @return random value (true or false)
   */
  public void generate(EvBinaryVectorIndividual[] pop, boolean[] computed) {

    if (!computed[index]) {
      for (int r = 0; r < parents.size(); r++) {
        if (!computed[parents.get(r).index]) {
          parents.get(r).generate(pop, computed);
        }
      }

      for (int i = 0; i < pop.length; i++) {
        if (probabilities.length == 1) {
          EvRandomizer.INSTANCE.nextInt(2);
          pop[i].setGene(index, EvRandomizer.INSTANCE
              .nextProbableBooleanAsInt(probabilities[0]));
        } else {

          int pos = 0;
          int x = 1;
          int parents_size = parents.size();

          for (int k = 0; k < parents_size; k++) {
            if (pop[i].getGene(parents.get(k).index) == 1) {
              pos += x;
            }
            x *= 2;
          }
          pop[i].setGene(index, EvRandomizer.INSTANCE
              .nextProbableBooleanAsInt(probabilities[pos]));

        }

      }
      computed[index] = true;
    }
  }


  /* ====================================================================================== */

  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("NODE [" + index + "]");
    builder.append("\nParents: ");
    for (EvBayesianNetworkNode parent : parents) {
      builder.append(parent.index + " ");
    }
    builder.append("\nChildren: ");
    for (EvBayesianNetworkNode child : children) {
      builder.append(child.index + " ");
    }
    builder.append("\n-\n");
    return builder.toString();
  }

}

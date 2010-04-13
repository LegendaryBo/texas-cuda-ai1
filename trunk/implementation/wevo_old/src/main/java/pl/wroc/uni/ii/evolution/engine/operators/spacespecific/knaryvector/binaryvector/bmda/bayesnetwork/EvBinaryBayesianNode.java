/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork;

import java.util.ArrayList;
import java.util.GregorianCalendar;
import java.util.List;

import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Jarek Fuks , Zbigniew Nazimek
 */
public class EvBinaryBayesianNode {
  private List<EvBinaryBayesianNode> parents;

  private double[] probabilities;

  private List<EvBinaryBayesianNode> children;

  private int index;

  private boolean regenerate;

  private int value;

  EvRandomizer r = new EvRandomizer(new GregorianCalendar().getTimeInMillis());


  public EvBinaryBayesianNode(double initial_probability, int index) {
    this.index = index;
    probabilities = new double[1];
    probabilities[0] = initial_probability;
    parents = new ArrayList<EvBinaryBayesianNode>();
    children = new ArrayList<EvBinaryBayesianNode>();
  }


  /**
   * @param i index of node
   */
  public EvBinaryBayesianNode(int i) {
    index = i;
    parents = new ArrayList<EvBinaryBayesianNode>();
    children = new ArrayList<EvBinaryBayesianNode>();
  }


  /**
   * set regenerate to false, is needed to generate new value
   */
  public void initGeneration() {
    regenerate = false;
  }


  /**
   * generate new value (when regenerate is false)
   * 
   * @return random value (1 or 0)
   */
  public int generate() {
    if (regenerate) {
      return value;
    }
    
    regenerate = true;

    if (probabilities.length == 1) {
      value = EvRandomizer.INSTANCE.nextProbableBooleanAsInt(probabilities[0]);
      return value;
    }
    int pos = 0;
    int x = 1;

    int parents_size = parents.size();
    int i = 0;
    for (i = 0; i < parents_size; i++) {
      if (parents.get(i).generate() == 1)
        pos += x;
      x *= 2;
    }

    return value =
        EvRandomizer.INSTANCE.nextProbableBooleanAsInt(probabilities[pos]);
  }


  /**
   * @return index of node
   */
  public int getIndex() {
    return index;
  }


  /**
   * add parent to node
   * 
   * @param node node, we want to be parent
   */
  public void addParent(EvBinaryBayesianNode node) {
    node.addChild(this);
    parents.add(node);
  }


  /**
   * @param node node, which we want to be remove from parents
   */
  public void removeParent(EvBinaryBayesianNode node) {
    node.removeChild(node);
    parents.remove(this);

  }


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


  protected void removeChild(EvBinaryBayesianNode child) {
    children.remove(child);
  }


  protected void addChild(EvBinaryBayesianNode child) {
    if (children.contains(child))
      throw new IllegalArgumentException("child already exist");
    children.add(child);
  }


  /**
   * @param descendant
   * @return true if param is decendant of current node
   */
  public boolean searchForDescendant(EvBinaryBayesianNode descendant) {
    if (children.size() > 0) {
      if (children.contains(descendant))
        return true;
      for (EvBinaryBayesianNode child : children) {
        if (child.searchForDescendant(descendant))
          return true;
      }
    }
    return false;
  }


  /**
   * @return parents of current node
   */
  public EvBinaryBayesianNode[] getParents() {
    EvBinaryBayesianNode[] ret = new EvBinaryBayesianNode[parents.size()];
    return ret = parents.toArray(ret);
  }


  /**
   * check if param is parent of current node
   * 
   * @param node
   * @return true if node is parent of this node
   */
  public boolean isParent(EvBinaryBayesianNode node) {
    return parents.contains(node);
  }


  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("NODE [" + index + "] <- ");

    for (EvBinaryBayesianNode parent : parents) {
      builder.append(parent.index + " ");
    }

    builder.append("\n");

    return builder.toString();
  }

}

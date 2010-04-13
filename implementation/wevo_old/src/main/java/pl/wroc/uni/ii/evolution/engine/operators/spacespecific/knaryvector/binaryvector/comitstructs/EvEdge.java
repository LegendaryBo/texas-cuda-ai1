package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs;

/**
 * Class representing Edge object containig Graph objects on its ends.
 * 
 * @author Kacper Gorski, Olgierd Humenczuk
 */
public class EvEdge implements Comparable {

  private int weight; // weight of the edge

  private EvGraph from_node = null; // Graph node one the one side

  private EvGraph to_node = null; // Graph node one the second side


  /**
   * Construcs Edge with given weight but with null Graph nodes
   * 
   * @param weight_ - of the current edge
   */
  public EvEdge(int weight_) {
    this.weight = weight_;
  }


  /**
   * @param weight_
   * @param from_node_
   * @param to_node_
   */
  public EvEdge(int weight_, EvGraph from_node_, EvGraph to_node_) {
    this.weight = weight_;
    this.from_node = from_node_;
    this.to_node = to_node_;
    this.from_node.addEdge(this);
    this.to_node.addEdge(this);
  }


  /**
   * Return Graph node adjecting to given Graph node through current edge.<BR>
   * If given node doesn't exists in Edge it returns the first Graph node in the
   * Edge
   * 
   * @param node - one of two nodes in Edge.
   * @return Graph node adjecting to given Graph node through current edge/
   */
  public EvGraph getIncNode(EvGraph node) {
    return node.equals(from_node) ? to_node : from_node;
  }


  /**
   * Set first node of this Edge to the one given as argument.
   * 
   * @param node replacing first node
   */
  public void setFromNode(EvGraph node) {
    if (to_node.equals(node))
      throw new IllegalArgumentException(
          "This node is already a to_node [cycles are forbidden]");
    from_node = node;
  }


  /**
   * Set first node of this Edge to the one given as argument.
   * 
   * @param node replacing first node
   */
  public void setToNode(EvGraph node) {
    if (from_node.equals(node))
      throw new IllegalArgumentException(
          "This node is already a from_node [cycles are forbidden]");
    to_node = node;
  }


  /**
   * Standard comparing function implemented from Comparable interface
   * 
   * @param arg Edge object to be compared
   * @return -1 if current weight is bigger.<BR>
   *         0 if weights are equal.<BR>
   *         1 if current weight is smaller.
   */
  public int compareTo(Object arg) {
    if (((EvEdge) arg).getWeight() < weight) {
      return -1;
    }
    if (((EvEdge) arg).getWeight() > weight) {
      return 1;
    }
    return 0;
  }


  /**
   * @param weight_ weight of current Edge object
   */
  public void setWeight(int weight_) {
    weight = weight_;
  }


  /**
   * @return the first node in current edge
   */
  public EvGraph getFromNode() {
    return from_node;
  }


  /**
   * @return the second node in current edge
   */
  public EvGraph getToNode() {
    return to_node;
  }


  /**
   * @return weight of current Edge object
   */
  public int getWeight() {
    return weight;
  }
}

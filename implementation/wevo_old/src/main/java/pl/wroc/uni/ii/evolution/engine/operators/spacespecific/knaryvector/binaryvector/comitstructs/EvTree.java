package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.comitstructs;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * Class representing single, labeled with int, node of a tree containig 2
 * double probability values.<BR>
 * <BR>
 * Used in generating popultion in commit algorithm
 * 
 * @author Kacper Gorski, Olgierd Humenczuk
 */
public class EvTree {

  private int label = 0;

  public int byl = 0;

  private ArrayList<EvTree> node_list = new ArrayList<EvTree>();

  // list of adjecting Tree nodes

  private EvTree parent = null; // parent of current node

  private double pbb_zero_if_zero = 0.0; // probability of zero if parent has

  // zero

  private double pbb_zero_if_one = 0.0; // probability of zero if parent has one


  /**
   * Constructs single Tree node with given label.<BR>
   * Probabilities are set to 0.0.
   * 
   * @param label of the node
   */
  public EvTree(int label) {
    this.label = label;
  }


  /**
   * Add given Tree node to current node's child list
   * 
   * @param child Tree node to add to child list
   */
  public void addChild(EvTree child) {
    node_list.add(child);
  }


  /**
   * Sets parrent of this object to given Tree node
   * 
   * @param parent Tree representing new parent of current object Put null if
   *        current tree node has to have no parent
   */
  public void setParent(EvTree parent) {
    this.parent = parent;
  }


  /**
   * @return parent of current Tree node.<br>
   *         returns null if there is no parent.
   */
  public EvTree getParent() {
    return parent;
  }


  /**
   * @return node_list - ArrayList of all children of this node
   */
  public ArrayList<EvTree> getChildren() {
    return node_list;
  }


  /**
   * @return label of this Tree object
   */
  public int getLabel() {
    return label;
  }


  /**
   * Sets pair of probability values in whole tree to which current Tree node
   * belongs too.<BR>
   * Function must be called to root of the tree to build up whole tree.<BR>
   * <BR>
   * Probability in root is always 0.5 and 0.5<BR>
   * Probability in rest of the Trees:<BR> - number of zeros on the same places
   * in adjecting individuals divided by overall number of zeros in parent
   * individual (0.0 if there are no zeros).<BR> - number of ones on the same
   * places in adjecting individuals divided by overall number of ones in parent
   * individual (0.0 if there are no ones).<BR>
   * 
   * @param population with individuals adjectings to nodes of the Tree.
   *        Population size must be the same as number of nodes of the tree.<BR>
   *        Tree node labeled with i int represent population[i] individual.<BR>
   */
  public void setProbability(EvPopulation<EvBinaryVectorIndividual> population) {
    ArrayList<EvTree> children = getChildren();

    EvBinaryVectorIndividual current_ind = population.get(label);
    EvBinaryVectorIndividual parent_ind = null;

    if (parent == null) { // default value for root
      pbb_zero_if_one = 0.5;
      pbb_zero_if_zero = 0.5;
    } else { // if not root
      int zero_zero = 0, zero_counter = 0, one_one = 0, one_counter = 0;

      parent_ind = population.get(this.parent.label);

      for (int i = 0; i < parent_ind.getDimension(); i++) { // for every child

        if (parent_ind.getGene(i) == 1) { // counting ones
          one_counter++;
          if (current_ind.getGene(i) == 0) {
            one_one++;
          }
        } else { // counting zeros
          zero_counter++;
          if (current_ind.getGene(i) == 0) {
            zero_zero++;
          }
        }

      }

      if (one_counter != 0) {
        pbb_zero_if_one = ((double) one_one) / ((double) one_counter);
      }
      if (zero_counter != 0) {
        pbb_zero_if_zero = ((double) zero_zero) / ((double) zero_counter);
      }
    }

    int children_size = children.size();

    for (int i = 0; i < children_size; i++) {
      children.get(i).setProbability(population); // recursivly to its children
    }

  }


  public double getPbb_zero_if_one() {
    return pbb_zero_if_one;
  }


  public void setPbb_zero_if_one(double pbb_zero_if_one) {
    this.pbb_zero_if_one = pbb_zero_if_one;
  }


  public double getPbb_zero_if_zero() {
    return pbb_zero_if_zero;
  }


  public void setPbb_zero_if_zero(double pbb_zero_if_zero) {
    this.pbb_zero_if_zero = pbb_zero_if_zero;
  }


  /**
   * Sets bits of all individuals given in table
   * 
   * @param ind_tab - table of individuals to be generated
   */
  public void randomizeIndividual(EvBinaryVectorIndividual[] ind_tab) {

    int tab_size = ind_tab.length;
    for (int i = 0; i < tab_size; i++) {
      EvBinaryVectorIndividual ind = ind_tab[i];

      if (parent == null) {
        if (Math.random() < pbb_zero_if_one) {
          ind.setGene(label, 0);
        } else {
          ind.setGene(label, 1);
        }
      } else {
        /** if parent has one */
        if (ind.getGene(parent.label) == 1) {
          if (Math.random() < pbb_zero_if_one)
            ind.setGene(label, 0);
          else
            ind.setGene(label, 1);

          /** if parent has zero */
        } else {
          if (Math.random() < pbb_zero_if_zero)
            ind.setGene(label, 0);
          else
            ind.setGene(label, 1);
        }
      }
    }

    int node_list_size = node_list.size();
    // TODO do iteration version of this
    for (int i = 0; i < node_list_size; i++) {
      node_list.get(i).randomizeIndividual(ind_tab);
    }

  }


  /**
   * 
   */
  public String toString() {
    return printTree(this);
  }


  /**
   * printing current Tree and its subTrees
   * 
   * @param
   * @return
   */
  private String printTree(EvTree t) {

    String ret = new String();

    ArrayList<EvTree> children = t.getChildren();
    ret += "node label:" + t.getLabel() +
    // "pbb_zero_if_zero" + t.pbb_zero_if_zero +
        // "pbb_zero_if_one" + t.pbb_zero_if_one +
        "\n";
    if (children.isEmpty())
      return ret;
    for (EvTree tree : children) {
      ret += "child: " + tree.label + "\n";
    }
    ret += "--------------------\n";

    if (t.byl == 0) {
      t.byl = 1;
      for (EvTree tree : children) {
        ret += printTree(tree);
      }
    } else
      ret = "STAAAAARE";

    return ret;
  }

}

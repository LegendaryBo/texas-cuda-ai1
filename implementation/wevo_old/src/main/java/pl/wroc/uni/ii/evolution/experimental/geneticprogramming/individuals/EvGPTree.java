package pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals;

import java.util.GregorianCalendar;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.EvGPTreeSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Zbigniew Nazimek, Donata Malecka
 */
public class EvGPTree extends EvIndividual {

  private static final long serialVersionUID = 9082966341126198039L;

  EvGPTree left, right; // right if type is unarny

  EvGPType type;

  double value;

  int index;

  EvRandomizer rand =
      new EvRandomizer(new GregorianCalendar().getTimeInMillis());


  public EvGPTree(EvGPType t, double v, int i) {
    type = t;
    value = v;
    index = i;

  }


  public int getHeight() {
    int height_left = 0;
    int height_right = 0;

    if (left != null)
      height_left = left.getHeight();
    if (right != null)
      height_right = right.getHeight();
    if (height_left > height_right)
      return height_left + 1;
    return height_right + 1;

  }


  public void mutate() {
    EvGPType new_type =
        EvGPType.values()[rand.nextInt(EvGPType.values().length - 1)];

    if (new_type == EvGPType.CONSTANT) {
      value = rand.nextDouble();
      left = null;
      right = null;
      type = new_type;
    }

    if (new_type == EvGPType.TABLE_ELEMENT) {
      index = rand.nextInt();
      left = null;
      right = null;
      type = new_type;
    } else if (new_type == EvGPType.COS || new_type == EvGPType.SIN
        || new_type == EvGPType.TAN) {
      left = null;
      EvGPTreeSolutionSpace sol = new EvGPTreeSolutionSpace(null);

      if (right == null) {
        right = sol.generateIndividual();
      }
      type = new_type;
    } else {
      EvGPTreeSolutionSpace sol = new EvGPTreeSolutionSpace(null);
      if (left == null) {
        left = sol.generateIndividual();
      }
      if (right == null) {
        right = sol.generateIndividual();
      }

      type = new_type;
    }

  }


  public boolean hasRight() {
    return (right != null);
  }


  public boolean hasLeft() {
    return (left != null);
  }


  public void setLeftSubTree(EvGPTree tree) {
    left = tree;
  }


  public void setRightSubTree(EvGPTree tree) {
    right = tree;
  }


  public EvGPTree getLeftSubTree() {
    return left;
  }


  public EvGPTree getRightSubTree() {
    return right;
  }


  public double eval(double[] x) {
    if (type == EvGPType.CONSTANT)
      return value;
    if (type == EvGPType.TABLE_ELEMENT) {
      if (x == null)
        return 0.0;
      if (index < 0)
        index = 0;
      if (index >= x.length)
        return x[x.length - 1];
      return x[index];
    }
    if (type == EvGPType.MUL)
      return left.eval(x) * right.eval(x);
    if (type == EvGPType.ADD)
      return left.eval(x) + right.eval(x);
    if (type == EvGPType.SUB)
      return left.eval(x) - right.eval(x);
    if (type == EvGPType.DIV) {
      double val = right.eval(x);
      if (val == 0)
        return 0; // / EXception ?????
      return left.eval(x) / val;
    }
    if (type == EvGPType.OR) {
      if ((left.eval(x) == 0.0) && (right.eval(x) == 0))
        return 0;
      return 1;
    }
    if (type == EvGPType.AND) {
      if ((left.eval(x) == 0.0) || (right.eval(x) == 0.0))
        return 0;
      return 1;
    }
    if (type == EvGPType.XOR) {
      if (((left.eval(x) == 0.0) && (right.eval(x) != 0))
          || (left.eval(x) != 0.0) && (right.eval(x) == 0))
        return 1;
      return 0;
    }
    if (type == EvGPType.COS)
      return Math.cos(right.eval(x));
    if (type == EvGPType.SIN)
      return Math.sin(right.eval(x));
    if (type == EvGPType.TAN)
      return Math.tan(right.eval(x));

    return 0;
  }


  @Override
  public Object clone() {
    EvGPTree node = new EvGPTree(type, value, index);
    if (left != null)
      node.left = (EvGPTree) left.clone();
    if (right != null)
      node.right = (EvGPTree) right.clone();

    return node;
  }

}

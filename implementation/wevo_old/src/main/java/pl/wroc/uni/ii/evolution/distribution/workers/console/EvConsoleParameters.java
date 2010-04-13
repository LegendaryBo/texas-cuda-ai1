package pl.wroc.uni.ii.evolution.distribution.workers.console;

import java.util.HashMap;

/**
 * Simple class pharsing application parameters given in console (in similar way
 * linux console does it).<br>
 * Example: input '-S http://127.0.0.1 -l logs.log -o output.out' should result
 * of 3 parameters:<br> - http://127.0.0.1 indexes by char 'S'<br> - logs.log
 * indexes by char 'l'<br> - output.out indexes by char 'o'<br>
 * <br>
 * Each parameter is indexed by single char letter (a-Z).
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvConsoleParameters {

  // Stored program parameters. Used by getParameter and parseParameters
  private HashMap<Character, String> parameters =
      new HashMap<Character, String>();


  /**
   * Pharse parameters and put them to a memory.<br>
   * To get their values call getParameter()
   * 
   * @param args - input of String array. Each object represents an index or a
   *        parameter.
   */
  public EvConsoleParameters(String[] args) {
    parseParameters(args);
  }


  /**
   * Return parameter indexed by <b>param</b>
   * 
   * @param param
   * @return String if parameter exists, null otherwise.
   */
  public String getParameter(char param) {
    return parameters.get(param);
  }


  /**
   * Tells if given parameter was parsed
   * 
   * @param param
   * @return true if parameter exists, false otherwise
   */
  public boolean parameterExist(char param) {
    if (parameters.get(param) == null) {
      return false;
    } else {
      return true;
    }
  }


  private void parseParameters(String[] args) {

    for (int i = 0; i < args.length - 1; i++) {
      if (isParam(args[i])) {
        parameters.put(getParam(args[i]), args[i + 1]);
        i++;
      }
    }

  }


  // tells if arg looks like "-%c" (%c is a char)
  private boolean isParam(String arg) {
    if (arg.length() != 2) {
      return false;
    } else {
      if (arg.charAt(0) == '-'
          && (arg.charAt(1) >= 'A' && arg.charAt(1) <= 'z')) {
        return true;
      } else {
        return false;
      }
    }
  }


  // when arg looks like "-%c" it returns %c
  private Character getParam(String arg) {
    return arg.charAt(1);
  }

}

#use this scrip to automaticaly add required jars to maven repository
mvn install:install-file -DgroupId=eg4eem -DartifactId=eg4eem -Dversion=1.0.0 -Dpackaging=jar -Dfile=eg4eem.jar

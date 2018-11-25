#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {
   int val;
   char str[20];
   
   strcpy(str, "98993489");
   val = atoi(str);
   printf("String value = %s, Int value = %d\n", str, val);

   strcpy(str, "tutorialspoint.com and 45");
   val = atoi(str);
   printf("String value = %s, Int value = %d\n", str, val);

   strcpy(str, "K");
   val = atoi(str);
   printf("String value = %s, Int value = %d\n", str, val);

   return(0);
}

/* From this we understand that atoi basically outputs the integer value from the string*/

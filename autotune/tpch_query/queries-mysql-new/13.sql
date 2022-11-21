select current_timestamp(6) into @query_start;
set @query_name='13.sql';
SELECT C_COUNT, COUNT(*) AS CUSTDIST FROM (SELECT C_CUSTKEY, COUNT(O_ORDERKEY) AS C_COUNT FROM CUSTOMER LEFT OUTER JOIN ORDERS ON C_CUSTKEY = O_CUSTKEY AND O_COMMENT NOT LIKE '%PENDING%DEPOSITS%' GROUP BY C_CUSTKEY) C_ORDERS GROUP BY C_COUNT ORDER BY CUSTDIST DESC, C_COUNT DESC;
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;